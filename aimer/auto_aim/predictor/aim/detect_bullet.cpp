#ifndef AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_CPP
#define AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_CPP

#include "detect_bullet.hpp"

#include <ctime>
#include <opencv2/imgproc.hpp>

#include "do_reproj.hpp"

namespace aimer::aim {
// 颜色判断参数
const float WEIGHTS[3] = { 4, 4, 2 };        // RGB三通道差异权重
const uint8_t DIFF_STEP = 5;                 // 帧差采样步长（减少计算量）
const uint8_t DIFF_THRESHOLD = 30;           // 帧差阈值（大于视为不同）
const cv::Size KERNEL1_SIZE = cv::Size(10, 10); // 形态学操作核1尺寸（用于膨胀）
const cv::Size KERNEL2_SIZE = cv::Size(4, 4);   // 形态学操作核2尺寸（用于开运算）
const cv::Scalar COLOR_LOWB = cv::Scalar(25, 40, 40);  // 弹丸颜色HSV范围下限
const cv::Scalar COLOR_UPB = cv::Scalar(90, 255, 255); // 弹丸颜色HSV范围上限
const cv::Scalar MIN_VUE = cv::Scalar(0, 255 * .1, 255 * .2); // 最低亮度/饱和度阈值

/* 颜色判断函数：判断像素是否属于弹丸颜色
 * @param hsv_col 待检测的HSV颜色值
 * @return bool  是否属于弹丸颜色
 * 判断条件：
 * 1. 亮度值V > 50（排除过暗区域）
 * 2. 色调H在50±(10+动态调整项)范围内（绿色系）
 * 动态调整项基于饱和度S和亮度V的均值进行指数缩放 */
bool test_is_bullet_color(const cv::Vec3b& hsv_col) {
    return hsv_col[2] > 50
        && fabs((int)hsv_col[0] - 50) < 10 + .5 * exp((hsv_col[1] + hsv_col[2]) / 100);
}

/* 帧差处理类：计算两帧之间的差异区域 */
class DoFrameDifference {
public:
    // 获取两帧差异区域（带形态学处理）
    cv::Mat get_diff(
        const cv::Mat& s1,       // 当前帧图像
        const cv::Mat& s2,       // 上一帧重投影图像
        const cv::Mat& ref,       // 颜色预筛选的参考区域
        const cv::Mat& lst_fr_bullets // 上一帧检测到的弹丸区域
    ) {
        this->tme -= (double)clock() / CLOCKS_PER_SEC;
        cv::Mat res = cv::Mat::zeros(s1.rows, s1.cols, CV_8U);
        
        // 步进采样遍历像素
        for (size_t y = 0; y < s1.rows; y += DIFF_STEP) {
            for (size_t x = 0; x < s1.cols; x += DIFF_STEP) {
                cv::Point p(x, y);
                // 跳过非参考区域和已检测区域
                if (!ref.at<uint8_t>(p) || (!lst_fr_bullets.empty() && lst_fr_bullets.at<uint8_t>(p)))
                    continue;
                
                // 计算当前点与周围点的加权颜色差异
                const cv::Vec3b& c1 = s1.at<cv::Vec3b>(p);
                bool flag = true;
                for (int dy = -0; dy < 1 && flag; ++dy) {
                    int ty = y + dy;
                    if (ty < 0 || ty >= s1.rows) continue;
                    for (int dx = -0; dx < 1 && flag; ++dx) {
                        int tx = x + dx;
                        if (tx < 0 || tx >= s1.cols) continue;
                        const cv::Vec3b& c2 = s2.at<cv::Vec3b>(cv::Point(tx, ty));
                        // 加权计算颜色差异
                        uint8_t tmp = (WEIGHTS[0] * abs(c1[0] - c2[0]) 
                                     + WEIGHTS[1] * abs(c1[1] - c2[1])
                                     + WEIGHTS[2] * abs(c1[2] - c2[2]))
                                     / (WEIGHTS[0] + WEIGHTS[1] + WEIGHTS[2]);
                        if (tmp < DIFF_THRESHOLD) flag = false;
                    }
                }
                res.at<uint8_t>(p) = flag ? 255 : 0;
            }
        }
        
        // 形态学膨胀处理（连接相邻区域）
        cv::dilate(res, res, this->kernel1);
        // 合并上一帧检测结果
        if (!lst_fr_bullets.empty()) res |= lst_fr_bullets;
        
        this->tme += (double)clock() / CLOCKS_PER_SEC;
        return res;
    }

private:
    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, KERNEL1_SIZE); // 膨胀核
    double tme = 0.; // 计时器
};

/* 弹丸检测主类 */
class DetectBullet {
public:
    DetectBullet() {
        // 初始化形态学处理核
        this->kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, KERNEL1_SIZE);
        this->kernel2 = cv::getStructuringElement(cv::MORPH_CROSS, KERNEL2_SIZE);
    }

    // 初始化重投影模块
    void init(const DoReproj& do_reproj) {
        this->do_reproj = do_reproj;
    }

    // 主处理函数：处理新帧并返回检测结果
    std::vector<ImageBullet> process_new_frame(
        const cv::Mat& new_frame,           // 输入的新帧图像
        const Eigen::Quaterniond& q         // 当前帧的姿态四元数
    ) {
        tme_total -= (double)clock() / CLOCKS_PER_SEC;
        
        // 更新帧缓存
        this->lst_hsv = this->cur_hsv.clone();
        this->lst_frame = this->cur_frame.clone();
        this->cur_frame = new_frame.clone();
        this->lst_fr_q = this->cur_fr_q;
        this->cur_fr_q = q;
        
        // 转换到HSV颜色空间
        cv::cvtColor(this->cur_frame, this->cur_hsv, cv::COLOR_BGR2HSV);
        
        // 首次运行不处理（需要前一帧数据）
        if (!this->lst_frame.empty()) {
            this->get_possible();  // 获取候选区域
            this->get_bullets();   // 筛选弹丸
        }
        
        tme_total += (double)clock() / CLOCKS_PER_SEC;
        return this->bullets;
    }

private:
    // 获取候选区域（颜色+帧差筛选）
    void get_possible() {
        // 重投影上一帧到当前视角
        cv::Mat lst_reproj = this->do_reproj.reproj(this->lst_hsv, this->lst_fr_q, this->cur_fr_q);
        
        // 颜色阈值筛选
        cv::Mat res, msk_not_dark;
        cv::inRange(this->cur_hsv, COLOR_LOWB, COLOR_UPB, res); // 颜色范围过滤
        cv::inRange(this->cur_hsv, MIN_VUE, cv::Scalar(255, 255, 255), msk_not_dark); // 排除过暗区域
        res &= msk_not_dark;
        
        // 帧差处理获取运动区域
        cv::Mat mat_diff = this->do_diff.get_diff(this->cur_hsv, lst_reproj, res, this->lst_msk);
        res &= mat_diff; // 交集作为候选
        
        // 形态学开运算去噪
        cv::morphologyEx(res, res, cv::MORPH_OPEN, this->kernel2);
        
        // 提取轮廓
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(res, this->contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    }

    // 轮廓点排序优化（按列排序，便于后续处理）
    void sort_points(std::vector<cv::Point>& vec) {
        // ...（实现按列排序逻辑，优化访问局部性）
    }

    // 判断轮廓是否包含弹丸特征像素
    bool test_is_bullet(std::vector<cv::Point> contour) {
        // 遍历轮廓内像素，检查是否存在符合弹丸颜色的点
        // ...（具体实现见原代码）
    }

    // 从候选轮廓中筛选弹丸
    void get_bullets() {
        this->bullets.clear();
        this->lst_msk = cv::Mat::zeros(this->cur_frame.rows, this->cur_frame.cols, CV_8U);
        
        for (uint32_t i = 0; i < this->contours.size(); ++i) {
            const auto& contour = this->contours[i];
            cv::RotatedRect rect = cv::minAreaRect(contour);
            cv::Size rect_size = rect.size;
            
            // 面积筛选（排除过小区域）
            if (rect_size.area() < 30) continue;
            
            // 轮廓填充率筛选（排除不规则区域）
            double ratio = cv::contourArea(contour) / rect_size.area();
            if (ratio < 0.5) continue;
            
            // 颜色特征验证
            if (this->test_is_bullet(contour)) {
                // 保存弹丸信息：中心坐标和半径
                this->bullets.emplace_back(
                    rect.center,
                    std::min(rect_size.height, rect_size.width) * .5
                );
                // 更新掩膜（用于下一帧处理）
                cv::drawContours(this->lst_msk, contours, i, 255, cv::FILLED);
            }
        }
    }

    // 成员变量
    cv::Mat cur_hsv, lst_hsv;          // 当前/上一帧的HSV图像
    cv::Mat cur_frame, lst_frame;      // 当前/上一帧的原始图像
    Eigen::Quaterniond cur_fr_q, lst_fr_q; // 当前/上一帧的姿态
    cv::Mat lst_msk;                   // 上一帧检测结果的掩膜
    std::vector<ImageBullet> bullets;   // 检测结果存储
    std::vector<std::vector<cv::Point>> contours; // 检测到的轮廓
    DoReproj do_reproj;                // 重投影模块
    DoFrameDifference do_diff;         // 帧差处理模块
    cv::Mat kernel1, kernel2;          // 形态学处理核
    // ...（其他计时变量等）
};
} // namespace aimer::aim

#endif /* AIMER_AUTO_AIM_PREDICTOR_AIM_DETECT_BULLET_CPP */