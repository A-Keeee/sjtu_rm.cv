#ifndef AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_CPP
#define AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_CPP

#include "detect_bullet.hpp"

#include <ctime>
#include <opencv2/imgproc.hpp>

#include "do_reproj.hpp"

namespace aimer::aim {

// 常量区：定义帧差计算权重、阈值，以及形态学操作核尺寸
const float WEIGHTS[3] = { 4, 4, 2 };        // RGB通道加权系数
const uint8_t DIFF_STEP = 5;                 // 帧差采样步长，减少计算量
const uint8_t DIFF_THRESHOLD = 30;           // 差分阈值，判定显著变化
const cv::Size KERNEL1_SIZE = cv::Size(10, 10);  // 膨胀操作核大小
const cv::Size KERNEL2_SIZE = cv::Size(4, 4);    // 开运算核大小
const cv::Scalar COLOR_LOWB = cv::Scalar(25, 40, 40);   // HSV低阈颜色
const cv::Scalar COLOR_UPB = cv::Scalar(90, 255, 255);  // HSV高阈颜色
const cv::Scalar MIN_VUE = cv::Scalar(0, 255 * .1, 255 * .2); // 最低亮度过滤

// 测试单像素是否满足“子弹颜色”特征：亮度足够，色相接近预设，并随饱和度/亮度做动态阈值
bool test_is_bullet_color(const cv::Vec3b& hsv_col) {
    return hsv_col[2] > 50
        && fabs((int)hsv_col[0] - 50) < 10 + .5 * exp((hsv_col[1] + hsv_col[2]) / 100);
}

// DoFrameDifference：基于 HSV 图像做帧差，并与原始掩码及历史子弹掩码取交
cv::Mat DoFrameDifference::get_diff(
    const cv::Mat& s1,
    const cv::Mat& s2,
    const cv::Mat& ref,
    const cv::Mat& lst_fr_bullets) {
    this->tme -= (double)clock() / CLOCKS_PER_SEC;
    // cv::imshow("s1", s1);
    // cv::imshow("s2", s2);
    cv::Mat res = cv::Mat::zeros(s1.rows, s1.cols, CV_8U);
    // 遍历图像，以步长 DIFF_STEP 采样点
    for (size_t y = 0; y < s1.rows; y += DIFF_STEP) {
        for (size_t x = 0; x < s1.cols; x += DIFF_STEP) {
            cv::Point p(x, y);
            if (!ref.at<uint8_t>(p) || (!lst_fr_bullets.empty() && lst_fr_bullets.at<uint8_t>(p)))
                continue;
            const cv::Vec3b& c1 = s1.at<cv::Vec3b>(p);
            bool flag = true;
            for (int dy = -0; dy < 1 && flag; ++dy) {
                int ty = y + dy;
                if (ty < 0 || ty >= s1.rows)
                    continue;
                for (int dx = -0; dx < 1 && flag; ++dx) {
                    int tx = x + dx;
                    if (tx < 0 || tx >= s1.cols)
                        continue;
                    const cv::Vec3b& c2 = s2.at<cv::Vec3b>(cv::Point(tx, ty));
                    uint8_t tmp = (WEIGHTS[0] * abs(c1[0] - c2[0]) + WEIGHTS[1] * abs(c1[1] - c2[1])
                                   + WEIGHTS[2] * abs(c1[2] - c2[2]))
                        / (WEIGHTS[0] + WEIGHTS[1] + WEIGHTS[2]);
                    if (tmp < DIFF_THRESHOLD)
                        flag = false;
                }
            }
            res.at<uint8_t>(p) = flag ? 255 : 0;
        }
    }
    // 膨胀操作填充小洞，再合并上一帧子弹区域，避免漏检
    cv::dilate(res, res, this->kernel1);
    if (!lst_fr_bullets.empty()) {
        res |= lst_fr_bullets;
    }
    // cv::imshow("diff", res);
    this->tme += (double)clock() / CLOCKS_PER_SEC;
    return res;
}

// DetectBullet 构造与初始化：创建形态学核并保存重投影对象
DetectBullet::DetectBullet() {
    this->kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, KERNEL1_SIZE); // 椭圆核
    this->kernel2 = cv::getStructuringElement(cv::MORPH_CROSS, KERNEL2_SIZE); // 十字核
}

void DetectBullet::init(const DoReproj& do_reproj) { //重投影
    this->do_reproj = do_reproj;
}

DetectBullet::DetectBullet(const DoReproj& do_reproj) {
    this->init(do_reproj);
}

// get_possible：筛选可能的子弹区域，流程如下：
// 1) 重投影上一帧 HSV 到当前视角  
// 2) HSV 范围过滤（色彩+亮度）  
// 3) 帧差过滤静止背景  
// 4) 开运算去噪，再找轮廓
void DetectBullet::get_possible() {
    tme_get_possible -= (double)clock() / CLOCKS_PER_SEC;

    // 对上一帧的 hsv 进行重投影
    assert(!this->lst_hsv.empty());
    cv::Mat lst_reproj = this->do_reproj.reproj(this->lst_hsv, this->lst_fr_q, this->cur_fr_q);

    cv::Mat res, msk_not_dark;
    // 先根据弹丸颜色判断可能是弹丸的部分
    // 在一个大概的绿色范围内
    cv::inRange(this->cur_hsv, COLOR_LOWB, COLOR_UPB, res);
    // 不能太黑
    cv::inRange(this->cur_hsv, MIN_VUE, cv::Scalar(255, 255, 255), msk_not_dark);
    res &= msk_not_dark;
    // cv::Mat tmp1 = res.clone();

    // 根据 hsv 作帧差
    cv::Mat mat_diff = this->do_diff.get_diff(this->cur_hsv, lst_reproj, res, this->lst_msk);
    // 如果上一帧某个位置有弹丸，需要考虑这一帧某颗弹丸跑到了同样的位置上
    res &= mat_diff;

    // cv::Mat show_mat = res + (tmp1 - res) * .5;
    // cv::imshow("show_mat", show_mat);

    // 进行滤波
    cv::morphologyEx(res, res, cv::MORPH_OPEN, this->kernel2);

    tme_get_possible += (double)clock() / CLOCKS_PER_SEC;

    // 寻找轮廓
    tme_find_contours -= (double)clock() / CLOCKS_PER_SEC;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(res, this->contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    tme_find_contours += (double)clock() / CLOCKS_PER_SEC;
    for (const cv::Vec4i& vc: hierarchy) {
        if (~vc[3]) {
            // std::cerr << "Nested contours!" << std::endl;
        }
    }
}

// sort_points：对轮廓点先按 X 分桶，再对每列内的 Y 值排序，生成有序点集
void DetectBullet::sort_points(std::vector<cv::Point>& vec) {
    tme_sort_points -= (double)clock() / CLOCKS_PER_SEC;

    if (this->sort_pts.empty()) {
        // 如果 sort_pts 没有初始化，那么先初始化
        this->sort_pts = std::vector<std::vector<uint32_t>>(this->cur_frame.cols);
    }

    uint32_t mn_x = this->cur_frame.cols, mx_x = 0;
    for (const cv::Point& pt: vec) {
        uint32_t x = pt.x, y = pt.y;
        this->sort_pts[x].emplace_back(y);
        if (x < mn_x)
            mn_x = x;
        if (x > mx_x)
            mx_x = x;
    }
    std::vector<cv::Point>().swap(vec);
    for (uint32_t x = mn_x; x <= mx_x; ++x) {
        std::vector<uint32_t>& vc_x = this->sort_pts[x];
        if (vc_x.size() > 10) {
            sort(vc_x.begin(), vc_x.end());
        } else {
            for (uint32_t i = 0; i < vc_x.size(); ++i) {
                for (uint32_t j = 0; j < i; ++j) {
                    if (vc_x[j] > vc_x[i])
                        std::swap(vc_x[i], vc_x[j]);
                }
            }
        }
        for (uint32_t y: vc_x) {
            vec.emplace_back(x, y);
        }
        std::vector<uint32_t>().swap(vc_x);
    }

    tme_sort_points += (double)clock() / CLOCKS_PER_SEC;
}

// test_is_bullet：在轮廓内部按列扫描，寻找最亮像素点并判断其是否符合子弹颜色
bool DetectBullet::test_is_bullet(std::vector<cv::Point> contour) {
    this->sort_points(contour);
    tme_get_brightest -= (double)clock() / CLOCKS_PER_SEC;

    this->sort_points(contour);
    bool flag = false;

    for (uint32_t i = 0, j = 0; i < contour.size() && !flag; i = j) {
        int x = contour[i].x;
        while (j < contour.size() && x == contour[j].x)
            ++j;
        assert(i < j);
        for (int y = contour[i].y; y <= contour[j - 1].y && !flag; ++y) {
            if (test_is_bullet_color(this->cur_hsv.at<cv::Vec3b>(cv::Point(x, y)))) {
                flag = true;
            }
        }
    }
    tme_get_brightest += (double)clock() / CLOCKS_PER_SEC;
    return flag;
}

// get_bullets：基于筛选轮廓，进一步按面积、长宽比和亮度检测，
// 并提取圆心与半径，结果保存到 bullets，同时生成下一帧遮罩 lst_msk
void DetectBullet::get_bullets() {
    bullets.clear();
    this->lst_msk = cv::Mat::zeros(this->cur_frame.rows, this->cur_frame.cols, CV_8U);

    // std::cerr << "contours size = " << this->contours.size() << std::endl;
    for (uint32_t i = 0; i < this->contours.size(); ++i) {
        const std::vector<cv::Point>& contour = this->contours[i];
        cv::RotatedRect rect = cv::minAreaRect(contour);
        cv::Size rect_size = rect.size;
        if (rect_size.area() < 30)
            continue;
        double ratio = cv::contourArea(contour) / rect_size.area();
        if (ratio < 0.5)
            continue;
        if (this->test_is_bullet(contour)) {
            this->bullets.emplace_back(
                rect.center,
                std::min(rect_size.height, rect_size.width) * .5);
            cv::drawContours(
                this->lst_msk,
                contours,
                i,
                255,
                cv::FILLED); // 在下一帧中作为上一帧识别到的子弹
        }
    }
}

// print_bullets：可视化检测结果，将每颗子弹画圆输出
cv::Mat DetectBullet::print_bullets() {
    cv::Mat res = this->cur_frame.clone();
    for (const ImageBullet& bul: this->bullets) {
        cv::circle(res, bul.center, bul.radius, cv::Scalar(255, 255, 255));
    }
    // cv::imshow("actual_res", res);
    // cv::waitKey(0);
    return res;
}

// process_new_frame：主入口，执行以下步骤：
// 1) 更新历史帧和四元数  
// 2) 转换当前帧到 HSV  
// 3) 调用 get_possible 筛选候选区域  
// 4) 调用 get_bullets 提取最终子弹信息  
std::vector<ImageBullet>
DetectBullet::process_new_frame(const cv::Mat& new_frame, const Eigen::Quaterniond& q) {
    tme_total -= (double)clock() / CLOCKS_PER_SEC;

    this->lst_hsv = this->cur_hsv.clone();
    this->lst_frame = this->cur_frame.clone();
    this->cur_frame = new_frame.clone();
    this->lst_fr_q = this->cur_fr_q;
    this->cur_fr_q = q;

    cv::cvtColor(this->cur_frame, this->cur_hsv, cv::COLOR_BGR2HSV);

    if (!this->lst_frame.empty()) {
        this->get_possible();
        this->get_bullets();
    }

    tme_total += (double)clock() / CLOCKS_PER_SEC;

    return this->bullets;
}

} // namespace aimer::aim

#endif /* AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_CPP */