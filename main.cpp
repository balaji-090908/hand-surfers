// HandSurfers: control Subway Surfers (or any arrow-key game) with your hand.
// Tracks the hand via skin-color segmentation (OpenCV only) and sends arrow keys.
//
// Controls (in the preview window):
//   c  - calibrate skin color: put your hand over the green box, then press c
//   q / ESC - quit
//   k  - toggle sending keys on/off (useful for testing)

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#elif defined(__APPLE__)
  #include <ApplicationServices/ApplicationServices.h>
#endif

enum class Key { Left, Right, Up, Down };

// ---------------------------------------------------------------- key sending
static void sendKey(Key k) {
#if defined(_WIN32)
    WORD vk = (k == Key::Left) ? VK_LEFT : (k == Key::Right) ? VK_RIGHT
            : (k == Key::Up)   ? VK_UP   : VK_DOWN;
    INPUT in[2] = {};
    in[0].type = INPUT_KEYBOARD;
    in[0].ki.wVk = vk;
    in[0].ki.dwFlags = KEYEVENTF_EXTENDEDKEY;
    in[1] = in[0];
    in[1].ki.dwFlags = KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP;
    SendInput(2, in, sizeof(INPUT));
#elif defined(__APPLE__)
    CGKeyCode code = (k == Key::Left) ? 123 : (k == Key::Right) ? 124
                   : (k == Key::Down) ? 125 : 126;
    CGEventRef down = CGEventCreateKeyboardEvent(nullptr, code, true);
    CGEventRef up   = CGEventCreateKeyboardEvent(nullptr, code, false);
    CGEventPost(kCGHIDEventTap, down);
    CGEventPost(kCGHIDEventTap, up);
    CFRelease(down);
    CFRelease(up);
#else
    // Linux (X11): needs `xdotool` installed (sudo apt install xdotool)
    const char* name = (k == Key::Left) ? "Left" : (k == Key::Right) ? "Right"
                     : (k == Key::Up)   ? "Up"   : "Down";
    std::string cmd = std::string("xdotool key ") + name + " &";
    int rc = system(cmd.c_str());
    (void)rc;
#endif
}

static const char* keyName(Key k) {
    switch (k) {
        case Key::Left:  return "LEFT";
        case Key::Right: return "RIGHT";
        case Key::Up:    return "UP (jump)";
        default:         return "DOWN (roll)";
    }
}

// ------------------------------------------- contour helpers (OpenCV-version independent)
static double polyArea(const std::vector<cv::Point>& p) {
    double a = 0;
    for (size_t i = 0, n = p.size(); i < n; ++i) {
        const cv::Point& p0 = p[i];
        const cv::Point& p1 = p[(i + 1) % n];
        a += double(p0.x) * p1.y - double(p1.x) * p0.y;
    }
    return std::abs(a) / 2.0;
}

static bool polyCentroid(const std::vector<cv::Point>& p, cv::Point& out) {
    double a = 0, cx = 0, cy = 0;
    for (size_t i = 0, n = p.size(); i < n; ++i) {
        const cv::Point& p0 = p[i];
        const cv::Point& p1 = p[(i + 1) % n];
        double cross = double(p0.x) * p1.y - double(p1.x) * p0.y;
        a  += cross;
        cx += (p0.x + p1.x) * cross;
        cy += (p0.y + p1.y) * cross;
    }
    if (std::abs(a) < 1e-6) return false;
    a *= 0.5;
    out = cv::Point(int(cx / (6.0 * a)), int(cy / (6.0 * a)));
    return true;
}

// ---------------------------------------------------------------- main
int main() {
    cv::VideoCapture cap;
    bool opened = false;
#if defined(_WIN32)
    // DirectShow is the most reliable backend for webcams on Windows; try a few camera indexes.
    for (int idx = 0; idx < 3 && !opened; ++idx) {
        opened = cap.open(idx, cv::CAP_DSHOW);
        if (opened) std::cout << "Opened camera " << idx << " (DirectShow)\n";
    }
#endif
    for (int idx = 0; idx < 3 && !opened; ++idx) {
        opened = cap.open(idx);
        if (opened) std::cout << "Opened camera " << idx << " (default backend)\n";
    }
    if (!opened) {
        std::cerr << "Could not open any webcam. Close other apps using the camera and check\n"
                     "Windows Settings > Privacy & security > Camera (allow desktop apps).\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Skin range in YCrCb (default works for many skin tones; press 'c' to calibrate)
    cv::Scalar lo(0, 133, 77), hi(255, 173, 127);

    bool sendKeys = true;
    bool calibrated = false;       // no gestures are sent until you press 'c'
    int tol = 10;                  // Cr/Cb tolerance around your skin color ( - / = to change )
    const int yTol = 70;           // brightness tolerance
    cv::Scalar calMean;
    auto applyRange = [&]() {
        lo = cv::Scalar(std::max(0.0, calMean[0] - yTol), calMean[1] - tol, calMean[2] - tol);
        hi = cv::Scalar(std::min(255.0, calMean[0] + yTol), calMean[1] + tol, calMean[2] + tol);
    };
    bool armed = true;   // must return to the neutral zone before the next gesture
    auto lastFire = std::chrono::steady_clock::now();
    const int cooldownMs = 350;
    std::string lastAction = "-";

    cv::Mat frame, ycrcb, mask;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Camera returned an empty frame. Exiting.\n";
            break;
        }
        cv::flip(frame, frame, 1);  // mirror so movement feels natural
        const int W = frame.cols, H = frame.rows;

        // Region of interest: right ~55% of the frame, so your face (left) is ignored.
        cv::Rect roi(int(W * 0.45), 0, W - int(W * 0.45), H);
        cv::Point roiCenter(roi.x + roi.width / 2, roi.y + roi.height / 2);
        // Neutral zone = box around the ROI center.
        cv::Rect neutral(roiCenter.x - roi.width / 6, roiCenter.y - roi.height / 6,
                         roi.width / 3, roi.height / 3);

        // Calibration box
        cv::Rect calBox(roiCenter.x - 20, roiCenter.y - 20, 40, 40);

        // Segment skin inside ROI
        cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);
        cv::inRange(ycrcb, lo, hi, mask);
        cv::Mat roiMask = cv::Mat::zeros(mask.size(), CV_8U);
        mask(roi).copyTo(roiMask(roi));
        cv::GaussianBlur(roiMask, roiMask, {7, 7}, 0);
        cv::threshold(roiMask, roiMask, 127, 255, cv::THRESH_BINARY);
        cv::morphologyEx(roiMask, roiMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(roiMask, roiMask, cv::MORPH_CLOSE, kernel);

        // Largest contour = hand
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(roiMask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        int best = -1;
        double bestArea = 3000;  // ignore tiny blobs
        for (size_t i = 0; i < contours.size(); ++i) {
            double a = polyArea(contours[i]);
            if (a > bestArea) { bestArea = a; best = (int)i; }
        }

        if (calibrated && best >= 0) {
            cv::Point c;
            if (polyCentroid(contours[best], c)) {
                cv::drawContours(frame, contours, best, {0, 255, 0}, 2);
                cv::circle(frame, c, 8, {0, 0, 255}, -1);

                if (neutral.contains(c)) {
                    armed = true;  // back at center: ready for next gesture
                } else if (armed) {
                    auto now = std::chrono::steady_clock::now();
                    auto since = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFire).count();
                    if (since >= cooldownMs) {
                        // Pick the direction with the largest overshoot out of the neutral box
                        int dx = 0, dy = 0;
                        if (c.x < neutral.x) dx = c.x - neutral.x;
                        else if (c.x > neutral.x + neutral.width) dx = c.x - (neutral.x + neutral.width);
                        if (c.y < neutral.y) dy = c.y - neutral.y;
                        else if (c.y > neutral.y + neutral.height) dy = c.y - (neutral.y + neutral.height);

                        Key k;
                        if (std::abs(dx) >= std::abs(dy)) k = (dx < 0) ? Key::Left : Key::Right;
                        else                              k = (dy < 0) ? Key::Up   : Key::Down;

                        if (sendKeys) sendKey(k);
                        lastAction = keyName(k);
                        lastFire = now;
                        armed = false;
                    }
                }
            }
        }

        // Overlay
        cv::rectangle(frame, roi, {255, 200, 0}, 1);
        cv::rectangle(frame, neutral, {255, 255, 255}, 2);
        cv::rectangle(frame, calBox, {0, 255, 0}, 1);
        cv::putText(frame, "Action: " + lastAction, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 255}, 2);
        cv::putText(frame, sendKeys ? "keys: ON (k)" : "keys: OFF (k)", {10, 50},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {200, 200, 200}, 1);
        cv::putText(frame, calibrated ? ("tolerance: " + std::to_string(tol) + "  (- smaller / = larger)")
                                      : std::string("NOT CALIBRATED: hand on green box, press c"),
                    {10, 75}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    calibrated ? cv::Scalar(200, 200, 200) : cv::Scalar(0, 0, 255), 1);
        cv::putText(frame, "c = calibrate | - = tighter | = looser | q = quit", {10, H - 10},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {200, 200, 200}, 1);

        cv::imshow("HandSurfers", frame);
        cv::imshow("Mask", roiMask);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == 'k') sendKeys = !sendKeys;
        if (key == 'c') {
            cv::Mat patch = ycrcb(calBox);
            cv::Scalar stddev;
            cv::meanStdDev(patch, calMean, stddev);
            calibrated = true;
            applyRange();
            std::cout << "Calibrated: Y=" << calMean[0] << " Cr=" << calMean[1] << " Cb=" << calMean[2] << "\n";
        }
        if (key == '-' && calibrated) {
            tol = std::max(3, tol - 2);
            applyRange();
        }
        if ((key == '=' || key == '+') && calibrated) {
            tol = std::min(40, tol + 2);
            applyRange();
        }
    }
    return 0;
}
