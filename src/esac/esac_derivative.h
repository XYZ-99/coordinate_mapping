#pragma once

#define PROB_THRESH 0.001

namespace esac {
    inline bool containsNaNs(const cv::Mat& m) {
        return cv::sum(cv::Mat(m != m))[0] > 0;
    }

    cv::Mat_<double> dProjectdObj(const cv::Point2f& pt, const cv::Point3f& obj, const cv::Mat& rot, const cv::Mat& trans, const cv::Mat& camMat, float maxReproErr) {
        double f = camMat.at<float>(0, 0);
        double ppx = camMat.at<float>(0, 2);
        double ppy = camMat.at<float>(1, 2);

        cv::Mat objMat = cv::Mat(obj);
        objMat.convertTo(objMat, CV_64F);

        objMat = rot * objMat + trans;

        if (std::abs(objMat.at<double>(2, 0)) < EPS)
            return cv::Mat_<double>::zeros(1, 3);

        double px = f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) + ppx;
        double py = f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) + ppy;

        double err = std::sqrt((pt.x - px) * (pt.x - px) + (pt.y - py) * (pt.y - py));

        if (err > maxReproErr)
            return cv::Mat_<double>::zeros(1, 3);

        err += EPS;

        double pxdx = f * rot.at<double>(0, 0) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 0);
        double pydx = f * rot.at<double>(1, 0) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 0);
        double dx = 0.5 / err * (2 * (pt.x - px) * - pxdx + 2 * (pt.y - py) * - pydx);

        double pxdy = f * rot.at<double>(0, 1) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 1);
        double pydy = f * rot.at<double>(1, 1) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 1);
        double dy = 0.5 / err * (2 * (pt.x - px) * - pxdy + 2 * (pt.y - py) * - pydy);

        double pxdz = f * rot.at<double>(0, 2) / objMat.at<double>(2, 0) - f * objMat.at<double>(0, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 2);
        double pydz = f * rot.at<double>(1, 2) / objMat.at<double>(2, 0) - f * objMat.at<double>(1, 0) / objMat.at<double>(2, 0) / objMat.at<double>(2, 0) * rot.at<double>(2, 2);
        double dz = 0.5 / err * (2 * (pt.x - px) * - pxdz + 2 * (pt.y - py) * - pydz);

        cv::Mat_<double> jacobean(1, 3);
        jacobean(0, 0) = dx;
        jacobean(0, 1) = dy;
        jacobean(0, 2) = dz;

        return jacobean;
    }

    cv::Mat_<double> dPNP(const std::vector<cv::Point2f>& image_points, std::vector<cv::Point3f> object_points, const cv::Mat& camMat, float eps = 0.001f) {
        int pnpMethod = (image_points.size() == 4) ? cv::SOLVEPNP_P3P : cv::SOLVEPNP_ITERATIVE;

        int effectiveObjPoints = (pnpMethod == cv::SOLVEPNP_P3P) ? 3 : object_points.size();

        cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(6, object_points.size() * 3);
        bool success;
        
        for (int i = 0; i < effectiveObjPoints; i++)
            for (unsigned j = 0; j < 3; j++) {
                if (j == 0) 
                    object_points[i].x += eps;
                else if (j == 1) 
                    object_points[i].y += eps;
                else if (j == 2) 
                    object_points[i].z += eps;

                esac::pose_t fStep;
                success = safeSolvePnP(object_points, image_points, camMat, cv::Mat(), fStep.first, fStep.second, false, pnpMethod);

                // forward
                if (!success)
                    return cv::Mat_<double>::zeros(6, object_points.size() * 3);

                if (j == 0) object_points[i].x -= 2 * eps;
                else if (j == 1) object_points[i].y -= 2 * eps;
                else if (j == 2) object_points[i].z -= 2 * eps;

                esac::pose_t bStep;
                success = safeSolvePnP(object_points, image_points, camMat, cv::Mat(), bStep.first, bStep.second, false, pnpMethod);

                // backward
                if (!success)
                    return cv::Mat_<double>::zeros(6, object_points.size() * 3);

                if (j == 0) object_points[i].x += eps;
                else if (j == 1) object_points[i].y += eps;
                else if (j == 2) object_points[i].z += eps;

                // gradient
                fStep.first = (fStep.first - bStep.first) / (2 * eps);
                fStep.second = (fStep.second - bStep.second) / (2 * eps);

                fStep.first.copyTo(jacobean.col(i * 3 + j).rowRange(0, 3));
                fStep.second.copyTo(jacobean.col(i * 3 + j).rowRange(3, 6));

                if (containsNaNs(jacobean.col(i * 3 + j)))
                    return cv::Mat_<double>::zeros(6, object_points.size() * 3);
            }

        return jacobean;
    }

    void dScore(esac::coord_t& sceneCoordinates, esac::hyp_assign_t& hypo_assignment, 
            const cv::Mat_<cv::Point2i>& sampling, const std::vector<std::vector<cv::Point2i> >& sampledPoints, 
            std::vector<cv::Mat_<double> >& jacobeansScore, const std::vector<double>& scoreOutputGradients, 
            const std::vector<esac::pose_t>& hyps, const std::vector<cv::Mat_<float> >& reproErrs, 
            const std::vector<cv::Mat_<double> >& jacobeansHyps, const std::vector<double>& hypProbs, 
            const cv::Mat& camMat, float inlierAlpha, float inlierBeta, float threshold, float maxReproErr)
    {
        int hypo_num = sampledPoints.size();
        
        // collect 2d-3D correspondences
        std::vector<std::vector<cv::Point2f> > image_points(hypo_num);
        std::vector<std::vector<cv::Point3f> > object_points(hypo_num);
        
        #pragma omp parallel for
        for (int h = 0; h < hypo_num; h++) {
            if (hypProbs[h] < PROB_THRESH) continue;

            int expert = hypo_assignment[h];

            for (unsigned i = 0; i < sampledPoints[h].size(); i++) {
                int x = sampledPoints[h][i].x;
                int y = sampledPoints[h][i].y;
          
                image_points[h].push_back(sampling(y, x));
                object_points[h].push_back(cv::Point3f(
                    sceneCoordinates[expert][0][y][x],
                    sceneCoordinates[expert][1][y][x],
                    sceneCoordinates[expert][2][y][x]));
            }
        }
        
        // derivatives of the soft inlier scores
        std::vector<cv::Mat_<double> > dReproErrs(reproErrs.size());

        #pragma omp parallel for
        for (int h = 0; h < hypo_num; h++) {
            if (hypProbs[h] < PROB_THRESH) 
                continue;
            
            dReproErrs[h] = cv::Mat_<double>::zeros(reproErrs[h].size());

            for (int x = 0; x < sampling.cols; x++)
                for (int y = 0; y < sampling.rows; y++) {
                    double softThreshold = inlierBeta * (reproErrs[h](y, x) - threshold);
                    softThreshold = 1 / (1+std::exp(-softThreshold));
                    dReproErrs[h](y, x) = -softThreshold * (1 - softThreshold) * inlierBeta * scoreOutputGradients[h];
                }

            dReproErrs[h] *= inlierAlpha  / dReproErrs[h].cols / dReproErrs[h].rows;
        }

        jacobeansScore.resize(hypo_num);

        // derivative of the loss
        #pragma omp parallel for
        for (int h = 0; h < hypo_num; h++) {
            cv::Mat_<double> jacobean = cv::Mat_<double>::zeros(1, sampling.cols * sampling.rows * 3);
            jacobeansScore[h] = jacobean;

            if (hypProbs[h] < PROB_THRESH) 
                continue;

            int expert = hypo_assignment[h];

            cv::Mat_<double> supportPointGradients = cv::Mat_<double>::zeros(1, 12);

            cv::Mat_<double> dHdO = dPNP(image_points[h], object_points[h], camMat);

            if (esac::getMax(dHdO) > 10) dHdO = 0;

            cv::Mat rot;
            cv::Rodrigues(hyps[h].first, rot);

            for (int x = 0; x < sampling.cols; x++)
                for (int y = 0; y < sampling.rows; y++) {
                    int ptIdx = x * dReproErrs[h].rows + y;

                    cv::Point2f pt(sampling(y, x).x, sampling(y, x).y);
                    cv::Point3f obj = cv::Point3f(
                        sceneCoordinates[expert][0][y][x],
                        sceneCoordinates[expert][1][y][x],
                        sceneCoordinates[expert][2][y][x]);
            
                    cv::Mat_<double> dPdO = dProjectdObj(pt, obj, rot, hyps[h].second, camMat, maxReproErr);
                    dPdO *= dReproErrs[h](y, x);
                    dPdO.copyTo(jacobean.colRange(x * dReproErrs[h].rows * 3 + y * 3, x * dReproErrs[h].rows * 3 + y * 3 + 3));

                    cv::Mat_<double> dPdH = jacobeansHyps[h].row(ptIdx);

                    supportPointGradients += dReproErrs[h](y, x) * dPdH * dHdO;
                }

            // accumulate the derivatives
            for (unsigned i = 0; i < sampledPoints[h].size(); i++) {
                unsigned x = sampledPoints[h][i].x;
                unsigned y = sampledPoints[h][i].y;
            
                jacobean.colRange(x * dReproErrs[h].rows * 3 + y * 3, x * dReproErrs[h].rows * 3 + y * 3 + 3) += supportPointGradients.colRange(i * 3, i * 3 + 3);
            }
        }
    }

    std::vector<cv::Mat_<double> > dSMScore(esac::coord_t& sceneCoordinates, esac::hyp_assign_t& hypo_assignment, 
            const cv::Mat_<cv::Point2i>& sampling, const std::vector<std::vector<cv::Point2i> >& sampledPoints, 
            const std::vector<double>& losses, const std::vector<double>& hypProbs, 
            const std::vector<esac::pose_t>& initHyps, const std::vector<cv::Mat_<float> >& initReproErrs, 
            const std::vector<cv::Mat_<double> >& jacobeansHyps, const cv::Mat& camMat, float inlierAlpha, 
            float inlierBeta, float threshold, float maxReproErr)
    {
        std::vector<double> scoreOutputGradients(sampledPoints.size());
            
        #pragma omp parallel for
        for (unsigned i = 0; i < sampledPoints.size(); i++) {
            if (hypProbs[i] < PROB_THRESH) 
                continue;

            scoreOutputGradients[i] = hypProbs[i] * losses[i];
            for (unsigned j = 0; j < sampledPoints.size(); j++)
                scoreOutputGradients[i] -= hypProbs[i] * hypProbs[j] * losses[j];
        }
     
        std::vector<cv::Mat_<double> > jacobeansScore;
        dScore(
            sceneCoordinates, 
            hypo_assignment,
            sampling, 
            sampledPoints, 
            jacobeansScore, 
            scoreOutputGradients, 
            initHyps, 
            initReproErrs, 
            jacobeansHyps, 
            hypProbs,
            camMat,
            inlierAlpha,
            inlierBeta,
            threshold,
            maxReproErr);

        #pragma omp parallel for
        for (unsigned i = 0; i < jacobeansScore.size(); i++) {
            cv::Mat_<double> reformat = cv::Mat_<double>::zeros(sampling.cols * sampling.rows, 3);
        
            if (hypProbs[i] >= PROB_THRESH) {
                for (int x = 0; x < sampling.cols; x++)
                    for (int y = 0; y < sampling.rows; y++) {
                        cv::Mat_<double> patchGrad = jacobeansScore[i].colRange(
                        x * sampling.rows * 3 + y * 3,
                        x * sampling.rows * 3 + y * 3 + 3);
                    
                        patchGrad.copyTo(reformat.row(y * sampling.cols + x));
                    }
            }

            jacobeansScore[i] = reformat;
        }
        
        return jacobeansScore;
    }
}
