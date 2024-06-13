#include "slam/map.h"
#include "slam/image_process.h"
#include "slam/config.h"
using namespace myslam;

Point2d ImageProcess:: pixel2cam(const Point2d& p, const Mat& K) {
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

// 特征点匹配
void ImageProcess::find_feature_matches(const Mat& img_1, const Mat& img_2, std::vector<KeyPoint>& keypoints_1,
                          std::vector<KeyPoint>& keypoints_2, std::vector<DMatch>& matches) 
{
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    // Hamming 距离
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配之间的最小距离和最大距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("Max : %f \n", max_dist);
    printf("Min : %f \n", min_dist);

    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}




// 使用Eigen计算3D-3D变换
Eigen::Isometry3d ImageProcess::estimateTransform(const vector<Point3f>& pts1, const vector<Point3f>& pts2) 
{
    // 计算中心
    Point3f p1, p2;
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 /= N;
    p2 /= N;

    // 去中心化
    vector<Point3f> q1(N), q2(N);
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // 计算W矩阵
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }

    // SVD分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Matrix3d R = U * (V.transpose());
    Eigen::Vector3d t = Eigen::Vector3d(p1.x, p1.y, p1.z) - R * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // 生成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(R);
    T.pretranslate(t);
    return T;
}

int ImageProcess::image_process(View& view) {
     ConfigReader config("../config/config.yaml");

        // 获取数据集路径
    string dataset_dir = config.getDatasetDir();
    string rgb_path = dataset_dir;
    string depth_path = dataset_dir;
    ifstream rgb_file(dataset_dir+"rgb.txt");
    ifstream depth_file(dataset_dir+"depth.txt");

 
    if (!rgb_file.is_open() || !depth_file.is_open()) {
        cerr << "无法打开文件" << endl;
        return -1;
    }

    string rgb_line, depth_line;
    vector<string> rgb_files, depth_files;
    while (getline(rgb_file, rgb_line) && getline(depth_file, depth_line)) {
        if (rgb_line[0] == '#' || depth_line[0] == '#') continue;  // 跳过注释行
        stringstream rgb_ss(rgb_line);
        stringstream depth_ss(depth_line);
        double rgb_time, depth_time;
        string rgb_file, depth_file;
        rgb_ss >> rgb_time >> rgb_file;
        depth_ss >> depth_time >> depth_file;
        rgb_files.push_back(rgb_path + rgb_file);
        depth_files.push_back(depth_path + depth_file);
    }

    for (size_t i = 0; i < rgb_files.size() - 1; ++i) {
        Mat img_1 = imread(rgb_files[i], IMREAD_COLOR);
        Mat img_2 = imread(rgb_files[i + 1], IMREAD_COLOR);

        if (img_1.empty() || img_2.empty()) {
            cerr << "无法读取图像" << endl;
            return -1;
        }

        vector<KeyPoint> keypoints_1, keypoints_2;
        vector<DMatch> matches;
        find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
        cout << matches.size() << "组匹配点" << endl;

        // 获取新增的特征点
        vector<KeyPoint> new_keypoints_2 = view.get_new_keypoints(keypoints_2, keypoints_1);

        // 绘制特征点和新增特征点
       
        view.draw_features(img_2, keypoints_2, new_keypoints_2);

        Mat d1 = imread(depth_files[i], IMREAD_UNCHANGED);
        Mat d2 = imread(depth_files[i + 1], IMREAD_UNCHANGED);

        vector<Point3f> pts_3d_1, pts_3d_2;
        for (DMatch m : matches) {
            ushort d_1 = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
            ushort d_2 = d2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
            if (d_1 == 0 || d_2 == 0)
                continue;
            float dd_1 = d_1 / 1000.0;
            float dd_2 = d_2 / 1000.0;
            Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
            Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
            pts_3d_1.push_back(Point3f(p1.x * dd_1, p1.y * dd_1, dd_1));
            pts_3d_2.push_back(Point3f(p2.x * dd_2, p2.y * dd_2, dd_2));
        }

        cout << "3d-3d : " << pts_3d_1.size() << endl;

        if (pts_3d_1.size() < 5 || pts_3d_2.size() < 5) {
            cerr << "Not enough points for pose estimation!" << endl;
            continue;
        }

        // 估计变换矩阵
        Eigen::Isometry3d pose = estimateTransform(pts_3d_1, pts_3d_2);
        cout << "Estimated pose:\n" << pose.matrix() << endl;

        // 累积位姿
        if (!poses.empty()) {
            pose = poses.back() * pose;
        }
        poses.push_back(pose);

        // 更新当前帧的3D特征点，转换到世界坐标系
        current_frame_points.clear();
        for (const auto& pt : pts_3d_2) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d transformed_point = pose * point;
            current_frame_points.push_back(Point3f(transformed_point.x(), transformed_point.y(), transformed_point.z()));
        }
    }
    return 0;
}