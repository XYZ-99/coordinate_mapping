#pragma once

#define EPS 0.00000001
#define PI 3.1415926

namespace esac
{
    cv::Mat_<cv::Point2i> create_sampling(
        unsigned outW, unsigned outH, 
        int subsampling, 
        int xshift, int yshift) 
    {
        cv::Mat_<cv::Point2i> sampling(outH, outW);

        #pragma omp parallel for
        for (unsigned x = 0; x < outW; x++)
            for (unsigned y = 0; y < outH; y++) {
                sampling(y, x) = cv::Point2i(
                    x * subsampling + subsampling / 2 - xshift,
                    y * subsampling + subsampling / 2 - yshift);
            }

        return sampling;
    }

    inline bool safeSolvePnP(const std::vector<cv::Point3f>& object_points, const std::vector<cv::Point2f>& image_points, 
        const cv::Mat& camMat, const cv::Mat& distCoeffs, cv::Mat& rot, 
        cv::Mat& trans, bool extrinsicGuess, int methodFlag) 
    {
        if (rot.type() == 0) 
            rot = cv::Mat_<double>::zeros(1, 3);
        if (trans.type() == 0) 
            trans= cv::Mat_<double>::zeros(1, 3);

        if (!cv::solvePnP(
            object_points, 
            image_points, 
            camMat, 
            distCoeffs, 
            rot, 
            trans, 
            extrinsicGuess,
            methodFlag)
        ) {
            rot = cv::Mat_<double>::zeros(3, 1);
            trans = cv::Mat_<double>::zeros(3, 1);
            return false;
        }

        return true;
    }

    inline void sampleHypotheses(esac::coord_t& scene_coords, esac::hyp_assign_t& hypo_assignment, 
            const cv::Mat_<cv::Point2i>& sampling, const cv::Mat_<float>& camMat, 
            unsigned maxSamplingTries, float threshold, std::vector<esac::pose_t>& hypotheses, 
            std::vector<std::vector<cv::Point2i>>& sampled_points, 
            std::vector<std::vector<cv::Point2f>>& image_points, 
            std::vector<std::vector<cv::Point3f>>& object_points)
    {
        int image_height = scene_coords.size(2);
        int image_width = scene_coords.size(3);
        int hypo_num = hypo_assignment.size(0);	

        sampled_points.resize(hypo_num);     
        image_points.resize(hypo_num);
        object_points.resize(hypo_num);
        hypotheses.resize(hypo_num);

        #pragma omp parallel for
        for (unsigned h = 0; h < hypotheses.size(); h++)
            for (unsigned t = 0; t < maxSamplingTries; t++) {
                int expert = hypo_assignment[h];

                std::vector<cv::Point2f> projections;
                cv::Mat_<uchar> alreadyChosen = cv::Mat_<uchar>::zeros(image_height, image_width);
                image_points[h].clear();
                object_points[h].clear();
                sampled_points[h].clear();

                for (int j = 0; j < 4; j++) {
                    // 2D location in the subsampled image
                    int x = irand(0, image_width-1);
                    int y = irand(0, image_height-1);

                    if (alreadyChosen(y, x) > 0) {
                        j--;
                        continue;
                    }

                    alreadyChosen(y, x) = 1;

                    // 2D location in the original RGB image
                    image_points[h].push_back(sampling(y, x)); 
                    // 3D object coordinate
                    object_points[h].push_back(cv::Point3f(
                        scene_coords[expert][0][y][x],
                        scene_coords[expert][1][y][x],
                        scene_coords[expert][2][y][x])); 
                    // 2D pixel location in the subsampled image
                    sampled_points[h].push_back(cv::Point2i(x, y)); 
                }

                if (!esac::safeSolvePnP(
                    object_points[h], 
                    image_points[h], 
                    camMat, 
                    cv::Mat(), 
                    hypotheses[h].first, 
                    hypotheses[h].second, 
                    false, 
                    cv::SOLVEPNP_P3P)
                ) {
                    continue;
                }

                cv::projectPoints(
                    object_points[h], 
                    hypotheses[h].first, 
                    hypotheses[h].second, 
                    camMat, 
                    cv::Mat(), 
                    projections
                );

                // check reconstruction, 4 sampled points should be reconstructed perfectly
                bool foundOutlier = false;
                for (unsigned j = 0; j < image_points[h].size(); j++) {
                    if (cv::norm(image_points[h][j] - projections[j]) < threshold)
                        continue;
                    foundOutlier = true;
                    break;
                }

                if (foundOutlier)
                    continue;
                else
                    break;
            }
    }

    inline std::vector<double> getHypScores(
        const std::vector<cv::Mat_<float>>& reproErrs,
        float threshold,
        float inlierAlpha,
        float inlierBeta)
    {
        std::vector<double> scores(reproErrs.size(), 0);

        #pragma omp parallel for
        for (unsigned h = 0; h < reproErrs.size(); h++)
            for (int x = 0; x < reproErrs[h].cols; x++)
                for (int y = 0; y < reproErrs[h].rows; y++) {
                    double softThreshold = inlierBeta * (reproErrs[h](y, x) - threshold);
                    softThreshold = 1 / (1+std::exp(-softThreshold));
                    scores[h] += 1 - softThreshold;
                }

        #pragma omp parallel for
        for (unsigned h = 0; h < reproErrs.size(); h++) {
            scores[h] *= inlierAlpha / reproErrs[h].cols / reproErrs[h].rows;
        }

        return scores;
    }

    cv::Mat_<float> getReproErrs(
        esac::coord_t& scene_coords,
        const esac::pose_t& hyp,
        int expert,
        const cv::Mat_<cv::Point2i>& sampling,
        const cv::Mat& camMat,
        float max_reprojection_error,
          cv::Mat_<double>& jacobeanHyp,
          bool calcJ = false)
    {
        cv::Mat_<float> reproErrs = cv::Mat_<float>::zeros(sampling.size());
        std::vector<cv::Point3f> points3D;
        std::vector<cv::Point2f> projections;	
        std::vector<cv::Point2f> points2D;
        std::vector<cv::Point2f> sources2D;

        for (int x = 0; x < sampling.cols; x++)
            for (int y = 0; y < sampling.rows; y++) {
                cv::Point2f pt2D(sampling(y, x).x, sampling(y, x).y);

                points3D.push_back(cv::Point3f(
                    scene_coords[expert][0][y][x],
                    scene_coords[expert][1][y][x],
                    scene_coords[expert][2][y][x]));
                points2D.push_back(pt2D);
                sources2D.push_back(cv::Point2f(x, y));
            }

        if (points3D.empty()) 
            return reproErrs;
        
        if (!calcJ) {
            cv::projectPoints(
                points3D, 
                hyp.first, 
                hyp.second, 
                camMat, 
                cv::Mat(), 
                projections);
        } else {
            cv::Mat_<double> projectionsJ;
            cv::projectPoints(
                points3D, 
                hyp.first, 
                hyp.second, 
                camMat, 
                cv::Mat(), 
                projections, 
                projectionsJ);

            projectionsJ = projectionsJ.colRange(0, 6);

            //assemble the jacobean of the refinement residuals
            jacobeanHyp = cv::Mat_<double>::zeros(points2D.size(), 6);
            cv::Mat_<double> dNdP(1, 2);
            cv::Mat_<double> dNdH(1, 6);

            for (unsigned ptIdx = 0; ptIdx < points2D.size(); ptIdx++) {
                double err = std::max(cv::norm(projections[ptIdx] - points2D[ptIdx]), EPS);
                if (err > max_reprojection_error)
                    continue;

                // derivative of norm
                dNdP(0, 0) = 1 / err * (projections[ptIdx].x - points2D[ptIdx].x);
                dNdP(0, 1) = 1 / err * (projections[ptIdx].y - points2D[ptIdx].y);

                dNdH = dNdP * projectionsJ.rowRange(2 * ptIdx, 2 * ptIdx + 2);
                dNdH.copyTo(jacobeanHyp.row(ptIdx));
            }
        }		

        // measure reprojection errors
        for (unsigned p = 0; p < projections.size(); p++) {
            cv::Point2f curPt = points2D[p] - projections[p];
            float l = std::min((float) cv::norm(curPt), max_reprojection_error);
            reproErrs(sources2D[p].y, sources2D[p].x) = l;
        }

        return reproErrs;    
    }

    inline void refineHyp(esac::coord_t& scene_coords, const cv::Mat_<float>& reproErrs, 
        const cv::Mat_<cv::Point2i>& sampling, const cv::Mat_<float>& camMat, 
        int expert, 
        float threshold, 
        unsigned maxRefSteps, float max_reprojection_error, esac::pose_t& hypothesis,
        cv::Mat_<int>& inlierMap)
    {
        cv::Mat_<float> localReproErrs = reproErrs.clone();
 
        unsigned bestInliers = 4; 

        for (unsigned rStep = 0; rStep < maxRefSteps; rStep++) {
            std::vector<cv::Point2f> localImgPts;
            std::vector<cv::Point3f> localObjPts; 
            cv::Mat_<int> localInlierMap = cv::Mat_<int>::zeros(localReproErrs.size());

            for (int x = 0; x < sampling.cols; x++)
                for (int y = 0; y < sampling.rows; y++) {
                    if (localReproErrs(y, x) < threshold) {
                        localImgPts.push_back(sampling(y, x));
                        localObjPts.push_back(cv::Point3f(
                            scene_coords[expert][0][y][x],
                            scene_coords[expert][1][y][x],
                            scene_coords[expert][2][y][x]));
                        localInlierMap(y, x) = 1;
                    }
                }

            if (localImgPts.size() <= bestInliers)
                break;
            bestInliers = localImgPts.size();

            esac::pose_t hypUpdate;
            hypUpdate.first = hypothesis.first.clone();
            hypUpdate.second = hypothesis.second.clone();

            if (!esac::safeSolvePnP(localObjPts, localImgPts, camMat, cv::Mat(), hypUpdate.first, hypUpdate.second, true, 
                (localImgPts.size() > 4) ? 
                    cv::SOLVEPNP_ITERATIVE : 
                    cv::SOLVEPNP_P3P)
            )
                break;

            hypothesis = hypUpdate;
            inlierMap = localInlierMap;

            cv::Mat_<double> jacobeanDummy;

            localReproErrs = esac::getReproErrs(
                scene_coords,
                hypothesis, 
                expert, 
                sampling, 
                camMat,
                max_reprojection_error,
                jacobeanDummy
            );
        }			
    }

    std::vector<double> softMax(const std::vector<double>& scores) {
        double maxScore = 0;
        for (unsigned i = 0; i < scores.size(); i++)
            if (i == 0 || scores[i] > maxScore) 
                maxScore = scores[i];

        std::vector<double> sf(scores.size());
        double sum = 0.0;

        for (unsigned i = 0; i < scores.size(); i++) {
            sf[i] = std::exp(scores[i] - maxScore);
            sum += sf[i];
        }
        for (unsigned i = 0; i < scores.size(); i++) {
            sf[i] /= sum;
        }

        return sf;
    }

    double entropy(const std::vector<double>& dist) {
        double e = 0;
        for (unsigned i = 0; i < dist.size(); i++)
            if (dist[i] > 0)
                e -= dist[i] * std::log2(dist[i]);

        return e;
    }

    int draw(const std::vector<double>& probs, bool training) {
        std::map<double, int> cumProb;
        double probSum = 0;
        double maxProb = -1;
        double maxIdx = 0; 

        for (unsigned idx = 0; idx < probs.size(); idx++) {
            if (probs[idx] < EPS) continue;

            probSum += probs[idx];
            cumProb[probSum] = idx;

            if (maxProb < 0 || probs[idx] > maxProb) {
                maxProb = probs[idx];
                maxIdx = idx;
            }
        }

        if (training)
            return cumProb.upper_bound(drand(0, probSum))->second;
        else
            return maxIdx;
    }

    esac::trans_t pose2trans(const esac::pose_t& pose) {
        esac::trans_t rot, trans = esac::trans_t::eye(4, 4);
        cv::Rodrigues(pose.first, rot);

        rot.copyTo(trans.rowRange(0,3).colRange(0,3));
        trans(0, 3) = pose.second.at<double>(0, 0);
        trans(1, 3) = pose.second.at<double>(1, 0);
        trans(2, 3) = pose.second.at<double>(2, 0);

        return trans.inv();
    }

    esac::pose_t trans2pose(const esac::trans_t& trans) {
        esac::trans_t invTrans = trans.inv();

        esac::pose_t pose;
        cv::Rodrigues(invTrans.colRange(0,3).rowRange(0,3), pose.first);

        pose.second = cv::Mat_<double>(3, 1);
        pose.second.at<double>(0, 0) = invTrans(0, 3);
        pose.second.at<double>(1, 0) = invTrans(1, 3);
        pose.second.at<double>(2, 0) = invTrans(2, 3);

        return pose; // camera transformation is inverted scene pose
    }

    double getAvg(const cv::Mat_<double>& mat) {
        double avg = 0;
        int count = 0;
        
        for (int x = 0; x < mat.cols; x++)
            for (int y = 0; y < mat.rows; y++) {
                double entry = std::abs(mat(y, x));
                if (entry > EPS) {
                    avg += entry;
                    count++;
                }
            }
        
        return avg / (EPS + count);
    }

    double getMax(const cv::Mat_<double>& mat) {
        double m = -1;
        
        for (int x = 0; x < mat.cols; x++)
            for (int y = 0; y < mat.rows; y++) {
                double val = std::abs(mat(y, x));
                if (m < 0 || val > m)
                    m = val;
            }
        
        return m;
    }

    double getMed(const cv::Mat_<double>& mat) {
        std::vector<double> vals;
        
        for (int x = 0; x < mat.cols; x++)
            for (int y = 0; y < mat.rows; y++) {
                double entry = std::abs(mat(y, x));
                if (entry > EPS)
                    vals.push_back(entry);
            }

        if (vals.empty()) 
            return 0;

        std::sort(vals.begin(), vals.end());
        
        return vals[vals.size() / 2];
    }	
}
