#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "thread_rand.h"
#include "stop_watch.h"

#include "esac_types.h"
#include "esac_util.h"
#include "esac_loss.h"
#include "esac_derivative.h"

#define MAX_SAMPLING_ATTEMPTS 1000000 
#define MAX_REF_STEPS 100 

int esac_forward(at::Tensor scene_coord_src, at::Tensor hypo_src, at::Tensor output_pose_src, 
                int xshift, int yshift,
                float focal_length,
                float ppointX, float ppointY,
                float threshold, float alpha, float beta, 
                float max_reprojection, int subsampling)
{
    esac::coord_t scene_coords = scene_coord_src.accessor<float, 4>();

    esac::hyp_assign_t hypo_assignment = hypo_src.accessor<long, 1>();

    int image_height = scene_coords.size(2);
    int image_width = scene_coords.size(3);
    int hypo_num = hypo_assignment.size(0);

    cv::Mat_<float> camera_matrix = cv::Mat_<float>::eye(3, 3);
    camera_matrix(0, 0) = focal_length;
    camera_matrix(1, 1) = focal_length;
    camera_matrix(0, 2) = ppointX;
    camera_matrix(1, 2) = ppointY;	

    cv::Mat_<cv::Point2i> sampling = 
        esac::create_sampling(image_width, image_height, subsampling, xshift, yshift);

    std::cout << "Sampling " << hypo_num << " hypotheses." << std::endl;
    StopWatch stopW;

    std::vector<esac::pose_t> hypotheses;
    std::vector<std::vector<cv::Point2i>> sampled_points;  
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::vector<cv::Point3f>> object_points;

    esac::sampleHypotheses(
        scene_coords,
        hypo_assignment,
        sampling,
        camera_matrix,
        MAX_SAMPLING_ATTEMPTS,
        threshold,
        hypotheses,
        sampled_points,
        image_points,
        object_points);

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << "Calculating scores." << std::endl;

    std::vector<cv::Mat_<float>> reproErrs(hypo_num);
    cv::Mat_<double> jacobeanDummy;

    #pragma omp parallel for 
    for (unsigned h = 0; h < hypotheses.size(); h++)
        reproErrs[h] = esac::getReproErrs(
        scene_coords,
        hypotheses[h], 
        hypo_assignment[h], 
        sampling, 
        camera_matrix,
        max_reprojection,
        jacobeanDummy
    );

    std::vector<double> scores = esac::getHypScores(
        reproErrs,
        threshold,
        alpha,
        beta
    );

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
    std::cout << "Drawing final hypothesis." << std::endl;	

    std::vector<double> hypProbs = esac::softMax(scores);
    double hypEntropy = esac::entropy(hypProbs);
    int hypIdx = esac::draw(hypProbs, false);

    std::cout << "Soft inlier count: " << scores[hypIdx] << " (Selection Probability: " << (int) (hypProbs[hypIdx]*100) << "%)" << std::endl; 
    std::cout << "Entropy of hypothesis distribution: " << hypEntropy << std::endl;


    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
    std::cout << "Refining winning pose:" << std::endl;

    cv::Mat_<int> inlierMap;

    esac::refineHyp(
        scene_coords,
        reproErrs[hypIdx],
        sampling,
        camera_matrix,
        hypo_assignment[hypIdx],
        threshold,
        MAX_REF_STEPS,
        max_reprojection,
        hypotheses[hypIdx],
        inlierMap
    );

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

    esac::trans_t estTrans = esac::pose2trans(hypotheses[hypIdx]);

    auto outPose = output_pose_src.accessor<float, 2>();
    for (unsigned x = 0; x < 4; x++)
    for (unsigned y = 0; y < 4; y++)
        outPose[y][x] = estTrans(y, x);	
        
    return hypo_assignment[hypIdx];
}

double esac_backward(at::Tensor scene_coord_src, at::Tensor outGradientsSrc, at::Tensor hypo_src,
        at::Tensor gt_pose_src, 
        float wLossRot, float wLossTrans, float lossCut,
        int xshift, int yshift,
        float focal_length,
        float ppointX,
        float ppointY,
        float threshold,
        float alpha, float beta, float max_reprojection, int subsampling)
{
    esac::coord_t scene_coords = scene_coord_src.accessor<float, 4>();

    esac::coord_t outGradients = outGradientsSrc.accessor<float, 4>();

    esac::hyp_assign_t hypo_assignment = hypo_src.accessor<long, 1>();

    int image_height = scene_coords.size(2);
    int image_width = scene_coords.size(3);
    int hypo_num = hypo_assignment.size(0);

    cv::Mat_<float> camera_matrix = cv::Mat_<float>::eye(3, 3);
    camera_matrix(0, 0) = focal_length;
    camera_matrix(1, 1) = focal_length;
    camera_matrix(0, 2) = ppointX;
    camera_matrix(1, 2) = ppointY;	

    esac::trans_t gtTrans(4, 4);
    auto gt_pose = gt_pose_src.accessor<float, 2>();

    for (unsigned x = 0; x < 4; x++)
        for (unsigned y = 0; y < 4; y++)
            gtTrans(y, x) = gt_pose[y][x];

    cv::Mat_<cv::Point2i> sampling = 
        esac::create_sampling(image_width, image_height, subsampling, xshift, yshift);

    std::cout << "Sampling " << hypo_num << " hypotheses." << std::endl;
    StopWatch stopW;

    std::vector<esac::pose_t> initHyps;
    std::vector<std::vector<cv::Point2i>> sampledPoints;  
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<std::vector<cv::Point3f>> object_points;

    esac::sampleHypotheses(
        scene_coords,
        hypo_assignment,
        sampling,
        camera_matrix,
        MAX_SAMPLING_ATTEMPTS,
        threshold,
        initHyps,
        sampledPoints,
        image_points,
        object_points
    );

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << "Calculating scores." << std::endl;

    // compute reprojection error images
    std::vector<cv::Mat_<float>> reproErrs(hypo_num);
    std::vector<cv::Mat_<double>> jacobeansHyp(hypo_num);

    #pragma omp parallel for 
    for (unsigned h = 0; h < initHyps.size(); h++)
        reproErrs[h] = esac::getReproErrs(
        scene_coords,
        initHyps[h], 
        hypo_assignment[h], 
        sampling, 
        camera_matrix,
        max_reprojection,
        jacobeansHyp[h],
        true
    );

    // soft inlier counting
    std::vector<double> scores = esac::getHypScores(
        reproErrs,
        threshold,
        alpha,
        beta
    );

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
    std::cout << "Drawing final hypothesis." << std::endl;	

    std::vector<double> hypProbs = esac::softMax(scores);
    double hypEntropy = esac::entropy(hypProbs);

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
    std::cout << "Refining poses:" << std::endl;

    std::vector<esac::pose_t> refHyps(hypo_num);
    std::vector<cv::Mat_<int>> inlierMaps(refHyps.size());
    
    #pragma omp parallel for
    for (unsigned h = 0; h < refHyps.size(); h++) {
        refHyps[h].first = initHyps[h].first.clone();
        refHyps[h].second = initHyps[h].second.clone();

        if (hypProbs[h] < PROB_THRESH) continue;

        esac::refineHyp(
            scene_coords,
            reproErrs[h],
            sampling,
            camera_matrix,
            hypo_assignment[h],
            threshold,
            MAX_REF_STEPS,
            max_reprojection,
            refHyps[h],
            inlierMaps[h]);
    }

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

    std::cout << "Entropy: " << hypEntropy << std::endl;

    // calculate expected pose loss
    double expectedLoss = 0;
    std::vector<double> losses(refHyps.size());

    for (unsigned h = 0; h < refHyps.size(); h++) {
        esac::trans_t estTrans = esac::pose2trans(refHyps[h]);
        losses[h] = esac::loss(estTrans, gtTrans, wLossRot, wLossTrans, lossCut);
        expectedLoss += hypProbs[h] * losses[h];
    }
    
    cv::Mat_<double> dLoss_dObj = cv::Mat_<double>::zeros(sampling.rows * sampling.cols, 3);

    std::cout << "Calculating gradients wrt hypotheses." << std::endl;

    std::vector<cv::Mat_<double>> dHyp_dObjs(refHyps.size());

    #pragma omp parallel for
    for (unsigned h = 0; h < refHyps.size(); h++) {
        int expert = hypo_assignment[h];

        dHyp_dObjs[h] = cv::Mat_<double>::zeros(6, sampling.rows * sampling.cols * 3);

        if (hypProbs[h] < PROB_THRESH) continue;

        std::vector<cv::Point2f> image_points;
        std::vector<cv::Point2i> srcPts;
        std::vector<cv::Point3f> object_points;

        for (int x = 0; x < inlierMaps[h].cols; x++)
            for (int y = 0; y < inlierMaps[h].rows; y++) {
                if (inlierMaps[h](y, x)) {
                    image_points.push_back(sampling(y, x));
                    srcPts.push_back(cv::Point2i(x, y));
                    object_points.push_back(cv::Point3f(
                        scene_coords[expert][0][y][x],
                        scene_coords[expert][1][y][x],
                        scene_coords[expert][2][y][x]));
                }
            }

        if (image_points.size() < 4)
            continue;

        std::vector<cv::Point2f> projections;
        cv::Mat_<double> projectionsJ;
        cv::projectPoints(object_points, refHyps[h].first, refHyps[h].second, camera_matrix, cv::Mat(), projections, projectionsJ);

        projectionsJ = projectionsJ.colRange(0, 6);

        cv::Mat_<double> jacobeanR = cv::Mat_<double> ::zeros(object_points.size(), 6);
        cv::Mat_<double> dNdP(1, 2);
        cv::Mat_<double> dNdH(1, 6);

        for (unsigned ptIdx = 0; ptIdx < object_points.size(); ptIdx++) {
            double err = std::max(cv::norm(projections[ptIdx] - image_points[ptIdx]), EPS);
            if (err > max_reprojection)
                continue;

            dNdP(0, 0) = 1 / err * (projections[ptIdx].x - image_points[ptIdx].x);
            dNdP(0, 1) = 1 / err * (projections[ptIdx].y - image_points[ptIdx].y);

            dNdH = dNdP * projectionsJ.rowRange(2 * ptIdx, 2 * ptIdx + 2);
            dNdH.copyTo(jacobeanR.row(ptIdx));
        }

        jacobeanR = - (jacobeanR.t() * jacobeanR).inv(cv::DECOMP_SVD) * jacobeanR.t();

        double maxJR = esac::getMax(jacobeanR);
        if (maxJR > 10) jacobeanR = 0;

        cv::Mat rot;
        cv::Rodrigues(refHyps[h].first, rot);

        for (unsigned ptIdx = 0; ptIdx < object_points.size(); ptIdx++) {
            cv::Mat_<double> dNdO = esac::dProjectdObj(image_points[ptIdx], object_points[ptIdx], rot, refHyps[h].second, camera_matrix, max_reprojection);
            dNdO = jacobeanR.col(ptIdx) * dNdO;

            int dIdx = srcPts[ptIdx].y * sampling.cols * 3 + srcPts[ptIdx].x * 3;
            dNdO.copyTo(dHyp_dObjs[h].colRange(dIdx, dIdx + 3));
        }
    }

    std::vector<cv::Mat_<double>> gradients(refHyps.size());
    esac::pose_t hypGT = esac::trans2pose(gtTrans);

    #pragma omp parallel for
    for (unsigned h = 0; h < refHyps.size(); h++) {
        if (hypProbs[h] < PROB_THRESH) 
            continue;

        cv::Mat_<double> dLoss_dHyp = esac::dLoss(refHyps[h], hypGT, wLossRot, wLossTrans, lossCut);
        gradients[h] = dLoss_dHyp * dHyp_dObjs[h];
    }

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

    std::cout << "Calculating gradients wrt scores." << std::endl;

    std::vector<cv::Mat_<double>> dLoss_dScore_dObjs = esac::dSMScore(
        scene_coords, 
        hypo_assignment,
        sampling, 
        sampledPoints, 
        losses, 
        hypProbs, 
        initHyps, 
        reproErrs, 
        jacobeansHyp,
        camera_matrix,
        alpha,
        beta,
        threshold,
        max_reprojection
    );

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

    for (unsigned h = 0; h < refHyps.size(); h++) {
        if (hypProbs[h] < PROB_THRESH) 
            continue;
        int expert = hypo_assignment[h];

        for (int idx = 0; idx < sampling.rows * sampling.cols; idx++) {
            int x = idx % sampling.cols;
            int y = idx / sampling.cols;
        
            outGradients[expert][0][y][x] += 
                hypProbs[h] * gradients[h](idx * 3 + 0) + dLoss_dScore_dObjs[h](idx, 0);
            outGradients[expert][1][y][x] += 
                hypProbs[h] * gradients[h](idx * 3 + 1) + dLoss_dScore_dObjs[h](idx, 1);
            outGradients[expert][2][y][x] += 
                hypProbs[h] * gradients[h](idx * 3 + 2) + dLoss_dScore_dObjs[h](idx, 2);
        }
    }

    return expectedLoss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &esac_forward, "ESAC forward");
    m.def("backward", &esac_backward, "ESAC backward");
}
