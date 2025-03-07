#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <math.h>
#include "algorithm"


using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

Mat Img;
void MainWindow::on_actionOPEN_FILE_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.jpg *.jpeg *.bmp)"));
    //filename = fileName;
    Mat img = cv::imread(fileName.toStdString(),IMREAD_GRAYSCALE);
    Img = cv::imread(fileName.toStdString(),IMREAD_GRAYSCALE);
    ui->label->setPixmap(
        QPixmap::fromImage(
            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8)
            )
        );



    Mat padded; //expand input image to optimal size
    int m = getOptimalDFTSize( img.rows );
    int n = getOptimalDFTSize( img.cols ); // on the border add zero values
    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI); // Add to the expanded another plane with zeros

    dft(complexI, complexI); // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    //fftshift(planes[0], planes[1]);
    Mat amplitude, angle;
    magnitude(planes[0], planes[1], amplitude);// planes[0] = magnitude


    phase(planes[0], planes[1], angle);
    angle.convertTo(angle, CV_8U);
    magnitude(planes[0], planes[1], planes[0]);
//--------------------------
    Mat magI = planes[0];

//    magI = abs(magI);
//    magI += Scalar::all(1); // switch to logarithmic scale
//    log(magI, magI);

//    float Fmax = *max_element(magI.begin<float>(), magI.end<float>());
//    float Fmin = *min_element(magI.begin<float>(), magI.end<float>());


//    log(1+abs(magI), magI);
//    magI = 255 * (magI - Fmin) / (Fmax - Fmin);
    //crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    //rearrange the quadrants of Fourier image so the origin is at image center
    int cx = amplitude.cols/2;
    int cy = amplitude.rows/2;

    Mat q0(amplitude, Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
    Mat q1(amplitude, Rect(cx, 0, cx, cy)); // Top-Right
    Mat q2(amplitude, Rect(0, cy, cx, cy)); // Bottom-Left
    Mat q3(amplitude, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    Mat amplitude_src;
    divide(amplitude, m*n, amplitude_src);

    amplitude += Scalar::all(1);
    log(amplitude, amplitude);

    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc(amplitude, &minVal, &maxVal, &minLoc, &maxLoc );
    amplitude = 255 * (amplitude - minVal) / (maxVal - minVal);

    normalize(amplitude, amplitude, 0, 255, NORM_MINMAX);
    amplitude.convertTo(amplitude, CV_8U);
    normalize(angle, angle, 0, 255, NORM_MINMAX);
    angle.convertTo(angle, CV_8U);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(angle.data, angle.cols, angle.rows, angle.step, QImage::Format_Grayscale8)
            )
        );
//-------------------------------
    Mat megI;
//    fftshift(planes[0], planes[1]);
    cv::merge(planes, 2, megI); // 实部与虚部合并
    cv::idft(megI, megI);       // idft结果也为复数
    megI = megI / megI.rows / megI.cols;
    cv::split(megI, planes);
    //planes[0] = planes[0]/255;
    //cv::idft(planes[0], planes[0]);
    //normalize(planes[0], planes[0], 0, 255, NORM_MINMAX);
    planes[0] += Scalar::all(1);
    log(planes[0], planes[0]);
    double minVal1;
    double maxVal1;
    Point minLoc1;
    Point maxLoc1;
    minMaxLoc(planes[0], &minVal1, &maxVal1, &minLoc1, &maxLoc1 );
    planes[0] = 255 * (planes[0] - minVal1) / (maxVal1 - minVal1);
    normalize(planes[0], planes[0], 0, 255, NORM_MINMAX);
    planes[0].convertTo(planes[0], CV_8U);

    megI = abs(megI);
    megI += Scalar::all(1);
    log(megI, megI);
//    double minVal12;
//    double maxVal12;
//    Point minLoc12;
//    Point maxLoc12;
//    minMaxLoc(megI, &minVal12, &maxVal12, &minLoc12, &maxLoc12 );
//    megI = 255 * (megI - minVal12) / (maxVal12 - minVal12);
    normalize(megI, megI, 0, 255, NORM_MINMAX);
    megI.convertTo(megI, CV_8U);

    cv::idft(complexI, complexI, DFT_INVERSE|DFT_SCALE|DFT_REAL_OUTPUT);
    complexI.convertTo(complexI, CV_8U);

    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(amplitude.data, amplitude.cols, amplitude.rows, amplitude.step, QImage::Format_Grayscale8)
            )
        );



    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(amplitude_src.data, amplitude_src.cols, amplitude_src.rows, amplitude_src.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(complexI.data, complexI.cols, complexI.rows, complexI.step, QImage::Format_Grayscale8)
            )
        );


//    Mat megI;
//    idft(planes[0], megI, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT|cv::DFT_SCALE);
//    megI.convertTo(megI, CV_8U);
//    normalize(megI, megI, 0, 255, NORM_MINMAX);

    ui->label_6->setPixmap(
        QPixmap::fromImage(
            QImage(planes[0].data, planes[0].cols, planes[0].rows, planes[0].step, QImage::Format_Grayscale8)
            )
        );
}

cv::Mat gaussian_low_pass_kernel(cv::Mat scr, float sigma);
cv::Mat gaussian_low_pass_filter(cv::Mat &src, float d0);
cv::Mat gaussian_high_pass_kernel(cv::Mat scr, float sigma);
cv::Mat gaussian_high_pass_filter(cv::Mat &src, float d0);


cv::Mat butterworth_low_kernel(cv::Mat &scr, float sigma, int n);

cv::Mat butterworth_low_pass_filter(cv::Mat &src, float d0, int n);

cv::Mat butterworth_high_kernel(cv::Mat &scr, float sigma, int n);

cv::Mat butterworth_high_pass_filter(cv::Mat &src, float d0, int n);

cv::Mat frequency_filter(cv::Mat &scr, cv::Mat &blur);

cv::Mat image_make_border(cv::Mat &src);

void fftshift(cv::Mat &plane0, cv::Mat &plane1);

void getcart(int rows, int cols, cv::Mat &x, cv::Mat &y);

Mat powZ(cv::InputArray src, double power);

Mat sqrtZ(cv::InputArray src);






void MainWindow::on_pushButton_clicked()
{
    Mat input = Img;

    int w = cv::getOptimalDFTSize(input.cols);
    int h = cv::getOptimalDFTSize(input.rows);
    cv::Mat padded;
    cv::copyMakeBorder(input, padded, 0, h - input.rows, 0, w - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    padded.convertTo(padded, CV_32FC1);

    for (int i = 0; i < padded.rows; i++)
    {
        float* ptr = padded.ptr<float>(i);
        for (int j = 0; j < padded.cols; j++)
            ptr[j] *= pow(-1, i + j);
    }

    cv::Mat plane[] = { padded,cv::Mat::zeros(padded.size(),CV_32F) };
    cv::Mat complexImg;
    cv::merge(plane, 2, complexImg);
    cv::dft(complexImg, complexImg);
    cv::split(complexImg, plane);
    cv::magnitude(plane[0], plane[1], plane[0]);
    plane[0] += cv::Scalar::all(1);
    cv::log(plane[0], plane[0]);
    cv::normalize(plane[0], plane[0], 1, 0, cv::NORM_MINMAX);

    cv::Mat idealBlur60(padded.size(), CV_32FC2);
    double D0_60 = 60;
    for (int i = 0; i < padded.rows; i++) {
        float* p = idealBlur60.ptr<float>(i);
        for (int j = 0; j < padded.cols; j++) {
            double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));
            if (d <= D0_60) {
                p[2 * j + 1] = 1;
                p[2 * j] = 1;
            }
            else {
                p[2 * j] = 0;
                p[2 * j + 1] = 0;
            }
        }
    }
    multiply(complexImg, idealBlur60, idealBlur60);
    cv::idft(idealBlur60, idealBlur60);
    cv::split(idealBlur60, plane);

    cv::magnitude(plane[0], plane[1], plane[0]);
    cv::normalize(plane[0], plane[0], 255, 0, cv::NORM_MINMAX);
    plane[0].convertTo(plane[0], CV_8U);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(plane[0].data, plane[0].cols, plane[0].rows, plane[0].step, QImage::Format_Grayscale8)
            )
        );


    cv::Mat idealBlur160(padded.size(), CV_32FC2);
    double D0_160 = 160;
    for (int i = 0; i < padded.rows; i++) {
        float* p = idealBlur160.ptr<float>(i);
        for (int j = 0; j < padded.cols; j++) {
            double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));
            if (d <= D0_160) {
                p[2 * j + 1] = 1;
                p[2 * j] = 1;
            }
            else {
                p[2 * j] = 0;
                p[2 * j + 1] = 0;
            }
        }
    }
    multiply(complexImg, idealBlur160, idealBlur160);
    cv::idft(idealBlur160, idealBlur160);
    cv::split(idealBlur160, plane);

    cv::magnitude(plane[0], plane[1], plane[0]);
    cv::normalize(plane[0], plane[0], 255, 0, cv::NORM_MINMAX);
    plane[0].convertTo(plane[0], CV_8U);

    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(plane[0].data, plane[0].cols, plane[0].rows, plane[0].step, QImage::Format_Grayscale8)
            )
        );

    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(input.data, input.cols, input.rows, input.step, QImage::Format_Grayscale8)
            )
        );

    cv::Mat idealBlurH60(padded.size(), CV_32FC2);
    double D0_H60 = 20;
    for (int i = 0; i < padded.rows; i++) {
        float* p = idealBlurH60.ptr<float>(i);
        for (int j = 0; j < padded.cols; j++) {
            double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));
            if (d <= D0_H60) {
                p[2 * j + 1] = 0;
                p[2 * j] = 0;
            }
            else {
                p[2 * j] = 1;
                p[2 * j + 1] = 1;
            }
        }
    }
    multiply(complexImg, idealBlurH60, idealBlurH60);
    cv::idft(idealBlurH60, idealBlurH60);
    cv::split(idealBlurH60, plane);


    cv::magnitude(plane[0], plane[1], plane[0]);
    cv::normalize(plane[0], plane[0], 255, 0, cv::NORM_MINMAX);
    plane[0].convertTo(plane[0], CV_8U);

    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(plane[0].data, plane[0].cols, plane[0].rows, plane[0].step, QImage::Format_Grayscale8)
            )
        );

    cv::Mat idealBlurH160(padded.size(), CV_32FC2);
    double D0_H160 = 120;
    for (int i = 0; i < padded.rows; i++) {
        float* p = idealBlurH160.ptr<float>(i);
        for (int j = 0; j < padded.cols; j++) {
            double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));//分子,计算pow必须为float型
            if (d <= D0_H160) {
                p[2 * j + 1] = 0;
                p[2 * j] = 0;
            }
            else {
                p[2 * j] = 1;
                p[2 * j + 1] = 1;
            }
        }
    }
    multiply(complexImg, idealBlurH160, idealBlurH160);
    cv::idft(idealBlurH160, idealBlurH160);
    cv::split(idealBlurH160, plane);


    cv::magnitude(plane[0], plane[1], plane[0]);
    cv::normalize(plane[0], plane[0], 255, 0, cv::NORM_MINMAX);
    plane[0].convertTo(plane[0], CV_8U);

    ui->label_6->setPixmap(
        QPixmap::fromImage(
            QImage(plane[0].data, plane[0].cols, plane[0].rows, plane[0].step, QImage::Format_Grayscale8)
            )
        );

}


void MainWindow::on_pushButton_2_clicked()
{
    Mat test = Img;

    float D0 = 50.0f;
    float D2 = 150.0f;
    float D1 = 20.0f;
    float D3 = 120.0f;

    Mat lowpass = butterworth_low_pass_filter(test, D0, 2);
    cv::normalize(lowpass, lowpass, 255, 0, cv::NORM_MINMAX);
    lowpass.convertTo(lowpass, CV_8U);
    Mat lowpass2 = butterworth_low_pass_filter(test, D2, 2);
    cv::normalize(lowpass2, lowpass2, 255, 0, cv::NORM_MINMAX);
    lowpass2.convertTo(lowpass2, CV_8U);
    Mat highpass = butterworth_high_pass_filter(test, D1, 2);
    cv::normalize(highpass, highpass, 255, 0, cv::NORM_MINMAX);
    highpass.convertTo(highpass, CV_8U);
    Mat highpass2 = butterworth_high_pass_filter(test, D3, 2);
    cv::normalize(highpass2, highpass2, 255, 0, cv::NORM_MINMAX);
    highpass2.convertTo(highpass2, CV_8U);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(lowpass.data, lowpass.cols, lowpass.rows, lowpass.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(lowpass2.data, lowpass2.cols, lowpass2.rows, lowpass2.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(test.data, test.cols, test.rows, test.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(highpass.data, highpass.cols, highpass.rows, highpass.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_6->setPixmap(
        QPixmap::fromImage(
            QImage(highpass2.data, highpass2.cols, highpass2.rows, highpass2.step, QImage::Format_Grayscale8)
            )
        );
}


void MainWindow::on_pushButton_3_clicked()
{
    Mat test = Img;

    float D0 = 50.0f;
    float D2 = 150.0f;
    float D1 = 20.0f;
    float D3 = 120.0f;

    Mat lowpass = gaussian_low_pass_filter(test, D0);
    cv::normalize(lowpass, lowpass, 255, 0, cv::NORM_MINMAX);
    lowpass.convertTo(lowpass, CV_8U);
    Mat lowpass2 = gaussian_low_pass_filter(test, D2);
    cv::normalize(lowpass2, lowpass2, 255, 0, cv::NORM_MINMAX);
    lowpass2.convertTo(lowpass2, CV_8U);
    Mat highpass = gaussian_high_pass_filter(test, D1);
    cv::normalize(highpass, highpass, 255, 0, cv::NORM_MINMAX);
    highpass.convertTo(highpass, CV_8U);
    Mat highpass2 = gaussian_high_pass_filter(test, D3);
    cv::normalize(highpass2, highpass2, 255, 0, cv::NORM_MINMAX);
    highpass2.convertTo(highpass2, CV_8U);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(lowpass.data, lowpass.cols, lowpass.rows, lowpass.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(lowpass2.data, lowpass2.cols, lowpass2.rows, lowpass2.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(test.data, test.cols, test.rows, test.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(highpass.data, highpass.cols, highpass.rows, highpass.step, QImage::Format_Grayscale8)
            )
        );

    ui->label_6->setPixmap(
        QPixmap::fromImage(
            QImage(highpass2.data, highpass2.cols, highpass2.rows, highpass2.step, QImage::Format_Grayscale8)
            )
        );

}

cv::Mat calc_Homomorphic(cv::Size size, float gamma_L, float gamma_H, float c, float DD)
{
    cv::Mat result(size, CV_32F);

    int cx = size.width / 2;
    int cy = size.height / 2;

    for (int i = 0; i < result.rows; ++i)
    {
        float* p = result.ptr<float>(i);
        for (int j = 0; j < result.cols; ++j)
        {
            float d_2 = std::pow(i - cy, 2) + std::pow(j - cx, 2);
            //同态滤波传递函数
            p[j] = (gamma_H - gamma_L) * (1 - std::exp(-c * d_2 / (DD * DD))) + gamma_L;
        }
    }

    //创建双通道图像，便于与复数图像相乘
    cv::Mat planes[] = { result.clone(), result.clone() };
    cv::Mat Homomorphic;
    merge(planes, 2, Homomorphic);

    return Homomorphic;
}

float gamma_L = 0.25, gamma_H = 2.0;
float c = 1.0, DD = 160;
void MainWindow::on_pushButton_4_clicked()
{
    Mat src = Img;
    //源图像转化为对数形式
    cv::Mat srclogMat;
    src.convertTo(srclogMat, CV_32F);
    srclogMat = srclogMat + cv::Scalar::all(1);
    cv::log(srclogMat, srclogMat);

    //扩展图像，扩展部分使用零填充
    cv::Mat paddedMat;
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);
    //on the border add zero values
    cv::copyMakeBorder(srclogMat, paddedMat, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat paddedMatdouble;
    paddedMat.convertTo(paddedMatdouble, CV_32F);
    //使傅里叶变换中心化
    Dftshift(paddedMatdouble, paddedMatdouble);

    //创建一个复数图像F(u, v)
    cv::Mat complexF;
    cv::dft(paddedMatdouble, complexF, cv::DFT_COMPLEX_OUTPUT);

    //创建同态滤波传递函数
    //double gamma_L = 0.25, gamma_H = 2.0;
    //double c = 1.0, D0 = 160;

    cv::Mat Homomorphic = calc_Homomorphic(paddedMat.size(), gamma_L, gamma_H, c, DD);

    cv::Mat complexFH;
    cv::Mat iDft;
    //采用对应像素相乘G(u, v) = H(u, v) * F(u, v)
    cv::multiply(complexF, Homomorphic, complexFH);
    cv::idft(complexFH, iDft, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    //最后不要忘了再完成一次移动
    Dftshift(iDft, iDft);

    //之前进行了对数变换，现在要变换回来
    cv::exp(iDft, iDft);
    iDft = iDft - cv::Scalar::all(1);
    cv::Mat result = iDft(cv::Rect(0, 0, src.cols, src.rows)).clone();
    //截断负值，归一化并转换为uchar
    MinusToZero(result, result);
    normalize(result, result, 0, 1, cv::NORM_MINMAX);
    result.convertTo(result, CV_8UC1, 255);


    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(result.data, result.cols, result.rows, result.step, QImage::Format_Grayscale8)
            )
        );
}


void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    gamma_H = position * 0.01;
    ui->label_10->setText(QString::number(position* 0.01));
}


void MainWindow::on_horizontalSlider_2_sliderMoved(int position)
{
    gamma_L = position * 0.01;
    ui->label_11->setText(QString::number(position * 0.01));
}


void MainWindow::on_horizontalSlider_3_sliderMoved(int position)
{
    DD = position;
    ui->label_12->setText(QString::number(position));
}



void MainWindow::on_pushButton_5_clicked()
{
    Mat gray_src = Img;
    //Mat src = imread("T.jpg");
    //Mat gray_src;
    //cvtColor(src, gray_src, CV_BGR2GRAY);
    Mat padded = Mat::zeros(2 * gray_src.rows, 2 * gray_src.cols, CV_32FC1);
    //图像扩充
    for (int row = 0; row < gray_src.rows; row++)
    {
        for (int col = 0; col < gray_src.cols; col++)
        {
            padded.at<float>(row, col) = gray_src.at<uchar>(row, col);
        }
    }
    //中心化
    for (int row = 0; row < padded.rows; row++)
    {
        for (int col = 0; col < padded.cols; col++)
        {
            padded.at<float>(row, col) *= pow(-1, row + col);
        }
    }
    //傅里叶变换
    Mat planes[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);
    //求原始图像的幅度谱和相位谱
    Mat temp1[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    split(complexImg, temp1);
    Mat amplitude1, angle1;
    amplitude1 = Magnitude(temp1[0], temp1[1]);
    angle1 = Phase(temp1[0], temp1[1]);
    //定义退化函数
    Mat degenerate[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    for (int row = 0; row < degenerate[0].rows; row++)
    {
        for (int col = 0; col < degenerate[0].cols; col++)
        {
            float v = M_PI*(0.1*(row - 262) + 0.1*(col - 175));
            if (v == 0)
            {
                degenerate[0].at<float>(row, col) = 1;
                degenerate[1].at<float>(row, col) = 0;
            }
            else
            {
                degenerate[0].at<float>(row, col) = sin(v) / v*cos(v);
                degenerate[1].at<float>(row, col) = -sin(v) / v*sin(v);
            }
        }
    }
    //求退化函数的幅度谱和相位谱
    Mat amplitude2, angle2;
    amplitude2 = Magnitude(degenerate[0], degenerate[1]);
    angle2 = Phase(degenerate[0], degenerate[1]);
    //图像退化
    Mat photo[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    for (int row = 0; row < padded.rows; row++)
    {
        for (int col = 0; col < padded.cols; col++)
        {
            photo[0].at<float>(row, col) = amplitude1.at<float>(row, col)*amplitude2.at<float>(row, col)*cos(angle1.at<float>(row, col) + angle2.at<float>(row, col));
            photo[1].at<float>(row, col) = amplitude1.at<float>(row, col)*amplitude2.at<float>(row, col)*sin(angle1.at<float>(row, col) + angle2.at<float>(row, col));
        }
    }
    //空域中显示退化图像
    Mat complex;
    merge(photo, 2, complex);
    idft(complex, complex);
    Mat temp2[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    split(complex, temp2);
    for (int row = 0; row < temp2[0].rows; row++)
    {
        for (int col = 0; col < temp2[0].cols; col++)
        {
            temp2[0].at<float>(row, col) *= pow(-1, row + col);
        }
    }
    for (int row = 0; row < temp2[0].rows; row++)
    {
        for (int col = 0; col < temp2[0].cols; col++)
        {
            if (temp2[0].at<float>(row, col) < 0)
                temp2[0].at<float>(row, col) = 0;
        }
    }
    normalize(temp2[0], temp2[0], 0, 255, cv::NORM_MINMAX);
    //temp2[0].convertTo(temp2[0], gray_src.type());
    temp2[0].convertTo(temp2[0], CV_8U);
    ui->label->setPixmap(
        QPixmap::fromImage(
            QImage(temp2[0].data, temp2[0].cols, temp2[0].rows, temp2[0].step, QImage::Format_Grayscale8)
            )
        );
    //imshow("退化图像", temp2[0]);
                               //添加高斯白噪声
    temp2[0] = addGaussianNoise(temp2[0]);
    temp2[0].convertTo(temp2[0], CV_8U);
    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(temp2[0].data, temp2[0].cols, temp2[0].rows, temp2[0].step, QImage::Format_Grayscale8)
            )
        );
  //imshow("添加了高斯白噪声的退化图像", temp2[0]);
    temp2[0].convertTo(temp2[0], CV_32FC1);
    //中心化
    for (int row = 0; row < temp2[0].rows; row++)
    {
        for (int col = 0; col < temp2[0].cols; col++)
        {
            temp2[0].at<float>(row, col) *= pow(-1, row + col);
        }
    }
    //求退化图像的傅里叶变换
    Mat picture[] = { Mat_<float>(temp2[0]),Mat::zeros(temp2[0].size(),CV_32FC1) };
    Mat img;
    merge(picture, 2, img);
    dft(img, img);
    //求退化图像的幅度谱和相位谱
    Mat temp3[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    split(img, temp3);
    Mat amplitude3, angle3;
    amplitude3 = Magnitude(temp3[0], temp3[1]);
    angle3 = Phase(temp3[0], temp3[1]);
    //定义维纳滤波函数
    Mat filter[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    for (int row = 0; row < filter[0].rows; row++)
    {
        for (int col = 0; col < filter[0].cols; col++)
        {
            float v = M_PI*(0.1*(row - 262) + 0.1*(col - 175));
            if (v == 0)
            {
                filter[0].at<float>(row, col) = amplitude2.at<float>(row, col) / (amplitude2.at<float>(row, col) + 0.08);
                filter[1].at<float>(row, col) = 0;
            }
            else
            {
                filter[0].at<float>(row, col) = v / sin(v)*cos(v)*amplitude2.at<float>(row, col) / (amplitude2.at<float>(row, col) + 0.08);
                filter[1].at<float>(row, col) = v*amplitude2.at<float>(row, col) / (amplitude2.at<float>(row, col) + 0.08);
            }
        }
    }
    //求维纳滤波函数的幅度谱和相位谱
    Mat amplitude4, angle4;
    amplitude4 = Magnitude(filter[0], filter[1]);
    angle4 = Phase(filter[0], filter[1]);
    //退化图像乘以维纳滤波函数
    Mat result[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    for (int row = 0; row < padded.rows; row++)
    {
        for (int col = 0; col < padded.cols; col++)
        {
            result[0].at<float>(row, col) = amplitude3.at<float>(row, col)*amplitude4.at<float>(row, col)*cos(angle3.at<float>(row, col) + angle4.at<float>(row, col));
            result[1].at<float>(row, col) = amplitude3.at<float>(row, col)*amplitude4.at<float>(row, col)*sin(angle3.at<float>(row, col) + angle4.at<float>(row, col));
        }
    }
    //图像复原
    Mat rst;
    merge(result, 2, rst);
    idft(rst, rst);
    Mat temp4[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32FC1) };
    split(rst, temp4);
    for (int row = 0; row < temp4[0].rows; row++)
    {
        for (int col = 0; col < temp4[0].cols; col++)
        {
            temp4[0].at<float>(row, col) *= pow(-1, row + col);
        }
    }
    for (int row = 0; row < temp4[0].rows; row++)
    {
        for (int col = 0; col < temp4[0].cols; col++)
        {
            if (temp4[0].at<float>(row, col) < 0)
                temp4[0].at<float>(row, col) = 0;
        }
    }
    normalize(temp4[0], temp4[0], 0, 255, cv::NORM_MINMAX);
    temp4[0].convertTo(temp4[0], CV_8U);
    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(temp4[0].data, temp4[0].cols, temp4[0].rows, temp4[0].step, QImage::Format_Grayscale8)
            )
        );
}

