#include "mainwindow.h"
#include "ui_mainwindow.h"

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
void MainWindow::on_actionopen_file_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.jpg *.jpeg *.bmp)"));
    Mat img = cv::imread(fileName.toStdString());
    Img = cv::imread(fileName.toStdString());
    cvtColor(img, img, COLOR_BGR2RGB);
    //QImage image = QImage((const unsigned char*)(image.data),image.cols,image.rows,QImage::Format_RGB888);
    ui->label->setPixmap(
        QPixmap::fromImage(
            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888)
            )
        );
}


void MainWindow::on_pushButton_clicked()
{
//    int i,j;
    Mat img = Img;
//    QImage img2(img.cols, img.rows, QImage::Format_RGB888);

//    for(i=0 ; i<img.rows ; i++)
//    {
//        for(j=0 ; j<img.cols ; j++)
//        {
//            // Set the pixel value of the QImage
//            //QImage img3 = img;
//            //QColor c = img.pixel(i,j);
//            int j2 = j + 10 * sin(j/10);

//            int r = img.at<Vec3b>(i,j2)[0];
//            int n = img.at<Vec3b>(i,j2)[1];
//            int m = img.at<Vec3b>(i,j2)[2];
//            img2.setPixel(j,i,qRgb(m,n,r));
//        }
//    }
    Mat img2,map_x,map_y;

    img2.create(img.size(),img.type());
    map_x.create(img.size(),CV_32FC1); //    map_x=Mat::zeros(srcImage.size(), CV_32FC1);
    map_y.create(img.size(),CV_32FC1);  //    map_y=Mat::zeros(srcImage.size(), CV_32FC1);

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j <img.cols; j++)
        {

            map_x.at<float>(i, j) = j + 10 * sin(i/10.0);
            map_y.at<float>(i, j) = i + 10 * sin(j/10.0);

        }
    }
    remap(img,img2,map_x,map_y,INTER_LINEAR,BORDER_CONSTANT,Scalar(0,0,0));

//    ui->label_2->setPixmap(QPixmap::fromImage(img2));
    cvtColor(img2, img2, COLOR_BGR2RGB);
    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(img2.data, img2.cols, img2.rows, img2.step, QImage::Format_RGB888)
            )
        );
}


void MainWindow::on_pushButton_2_clicked()
{

    Mat src = Img;
    Mat img2;
    Point2f srcPoints[4];
    Point2f dstPoints[4];

    srcPoints[0] = Point2f(0, 0);
    srcPoints[1] = Point2f(0, src.rows);
    srcPoints[2] = Point2f(src.cols, 0);
    srcPoints[3] = Point2f(src.cols, src.rows);

    dstPoints[0] = Point2f(0, 0);
    dstPoints[1] = Point2f(src.cols*0.2, src.rows*0.8);
    dstPoints[2] = Point2f(src.cols, 0);
    dstPoints[3] = Point2f(src.cols*0.8, src.rows*0.8);
    Mat M1 = getPerspectiveTransform(srcPoints, dstPoints);
    warpPerspective(src, img2, M1, src.size());


    //    ui->label_2->setPixmap(QPixmap::fromImage(img2));
    cvtColor(img2, img2, COLOR_BGR2RGB);
    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(img2.data, img2.cols, img2.rows, img2.step, QImage::Format_RGB888)
            )
        );
}


void MainWindow::on_pushButton_3_clicked()
{
    Mat src = Img;
    Mat dst;

    //Point2f center=Point2f(img.cols/2,img.rows/2);
    //warpPolar(img,img1,Size(300,600),center,center.x,INTER_LINEAR+WARP_POLAR_LINEAR);
    //warpPolar(img,img2,Size(img.rows,img.cols),center,center.x,INTER_LINEAR+WARP_POLAR_LINEAR+WARP_INVERSE_MAP);

    //Mat img2,map_x,map_y;

    dst.create(src.size(),src.type());
    //map_x.create(img.size(),CV_32FC1); //    map_x=Mat::zeros(srcImage.size(), CV_32FC1);
    //map_y.create(img.size(),CV_32FC1);  //    map_y=Mat::zeros(srcImage.size(), CV_32FC1);


    for (int id = 0; id < dst.rows; id++) {
        for (int jd = 0; jd < dst.cols; jd++) {
            double xd = jd * 2.0 / dst.cols - 1.0;
            double yd = id * 2.0 / dst.cols - 1.0;
            double rd = sqrt(xd * xd + yd * yd);
            double phid = atan2(yd, xd);
            double xs = asin(rd) * 2 / M_PI * cos(phid);
            double ys = asin(rd) * 2 / M_PI * sin(phid);
            int is = round((ys + 1.0) * dst.rows / 2.0);
            int js = round((xs + 1.0) * dst.cols / 2.0);
            if (is > dst.rows || is < 0 || js>dst.cols || js < 0)
                continue;
            dst.at<Vec3b>(id, jd)[0] = src.at<Vec3b>(is, js)[0];
            dst.at<Vec3b>(id, jd)[1] = src.at<Vec3b>(is, js)[1];
            dst.at<Vec3b>(id, jd)[2] = src.at<Vec3b>(is, js)[2];
        }
    }

    cvtColor(dst, dst, COLOR_BGR2RGB);

//    int wc = ui->label_2->width();
//    int hc = ui->label_2->height();
//    cv::resize(dst, dst, Size(wc, hc));
    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(dst.data, dst.cols, dst.rows, dst.step, QImage::Format_RGB888)
            )
        );

}

class WaveTransform
{
public:
    Mat WDT(const Mat &_src, const string _wname, const int _level);//小波分解
    Mat IWDT(const Mat &_src, const string _wname, const int _level);//小波重构
    void wavelet_D(const string _wname, Mat &_lowFilter, Mat &_highFilter);//分解包
    void wavelet_R(const string _wname, Mat &_lowFilter, Mat &_highFilter);//重构包
    Mat waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
    Mat waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);
};
Mat WaveTransform::WDT(const Mat &_src, const string _wname, const int _level)
{
    Mat_<float> src = Mat_<float>(_src);
    Mat dst = Mat::zeros(src.rows, src.cols, src.type());
    int row = src.rows;
    int col = src.cols;
    //高通低通滤波器
    Mat lowFilter;
    Mat highFilter;
    wavelet_D(_wname, lowFilter, highFilter);
    //小波变换
    int t = 1;

    while (t <= _level)
    {
        //先进行 行小波变换
        //#pragma omp parallel for
        for (int i = 0;i<row;i++)
        {
            //取出src中要处理的数据的一行
            Mat oneRow = Mat::zeros(1, col, src.type());

            for (int j = 0;j<col;j++)
            {
                oneRow.at<float>(0, j) = src.at<float>(i, j);
            }

            oneRow = waveletDecompose(oneRow, lowFilter, highFilter);
            for (int j = 0;j<col;j++)
            {
                dst.at<float>(i, j) = oneRow.at<float>(0, j);
            }
        }

        //小波列变换
        //#pragma omp parallel for
        for (int j = 0;j<col;j++)
        {
            Mat oneCol = Mat::zeros(row, 1, src.type());

            for (int i = 0;i<row;i++)
            {
                oneCol.at<float>(i, 0) = dst.at<float>(i, j);//dst,not src
            }
            oneCol = (waveletDecompose(oneCol.t(), lowFilter, highFilter)).t();

            for (int i = 0;i<row;i++)
            {
                dst.at<float>(i, j) = oneCol.at<float>(i, 0);
            }
        }
        //更新
        row /= 2;
        col /= 2;
        t++;
        src = dst;
    }
    return dst;
}

Mat WaveTransform::IWDT(const Mat &_src, const string _wname, const int _level)
{

    Mat src = Mat_<float>(_src);
    Mat dst;
    src.copyTo(dst);
    int N = src.rows;
    int D = src.cols;

    //高低通滤波器
    Mat lowFilter;
    Mat highFilter;
    wavelet_R(_wname, lowFilter, highFilter);

    //小波变换
    int t = 1;
    int row = N / std::pow(2., _level - 1);
    int col = D / std::pow(2., _level - 1);

    while (row <= N && col <= D)
    //while(t<=_level)
    {
        //列逆变换
        for (int j = 0;j<col;j++)
        {
            Mat oneCol = Mat::zeros(row, 1, src.type());

            for (int i = 0;i<row;i++)
            {
                oneCol.at<float>(i, 0) = src.at<float>(i, j);
            }
            oneCol = (waveletReconstruct(oneCol.t(), lowFilter, highFilter)).t();

            for (int i = 0;i<row;i++)
            {
                dst.at<float>(i, j) = oneCol.at<float>(i, 0);
            }

        }

        //行逆变换
        for (int i = 0;i<row;i++)
        {
            Mat oneRow = Mat::zeros(1, col, src.type());
            for (int j = 0;j<col;j++)
            {
                oneRow.at<float>(0, j) = dst.at<float>(i, j);
            }
            oneRow = waveletReconstruct(oneRow, lowFilter, highFilter);
            for (int j = 0;j<col;j++)
            {
                dst.at<float>(i, j) = oneRow.at<float>(0, j);
            }
        }

        row *= 2;
        col *= 2;
        t++;
        src = dst;
    }

    return dst;
}


void WaveTransform::wavelet_D(const string _wname, Mat &_lowFilter, Mat &_highFilter)
{
    if (_wname == "haar" || _wname == "db1")
    {
        int N = 2;
        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        _lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
        _lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

        _highFilter.at<float>(0, 0) = -1 / sqrtf(N);
        _highFilter.at<float>(0, 1) = 1 / sqrtf(N);
    }
    else if (_wname == "sym2")
    {
        int N = 4;
        float h[] = { -0.4830, 0.8365, -0.2241, -0.1294 };
        float l[] = { -0.1294, 0.2241,  0.8365, 0.4830 };

        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        for (int i = 0;i<N;i++)
        {
            _lowFilter.at<float>(0, i) = l[i];
            _highFilter.at<float>(0, i) = h[i];
        }
    }
}
void WaveTransform::wavelet_R(const string _wname, Mat &_lowFilter, Mat &_highFilter)
{
    if (_wname == "haar" || _wname == "db1")
    {
        int N = 2;
        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);


        _lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
        _lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

        _highFilter.at<float>(0, 0) = 1 / sqrtf(N);
        _highFilter.at<float>(0, 1) = -1 / sqrtf(N);
    }
    else if (_wname == "sym2")
    {
        int N = 4;
        float h[] = { -0.1294,-0.2241,0.8365,-0.4830 };
        float l[] = { 0.4830, 0.8365, 0.2241, -0.1294 };

        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        for (int i = 0;i<N;i++)
        {
            _lowFilter.at<float>(0, i) = l[i];
            _highFilter.at<float>(0, i) = h[i];
        }
    }
}


Mat WaveTransform::waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter)
{
    assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
    assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
    Mat src = Mat_<float>(_src);

    int D = src.cols;

    Mat lowFilter = Mat_<float>(_lowFilter);
    Mat highFilter = Mat_<float>(_highFilter);

    //频域滤波或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter)
    Mat dst1 = Mat::zeros(1, D, src.type());
    Mat dst2 = Mat::zeros(1, D, src.type());

    filter2D(src, dst1, -1, lowFilter);
    filter2D(src, dst2, -1, highFilter);

    //下采样
    //数据拼接
    for (int i = 0, j = 1;i<D / 2;i++, j += 2)
    {
        src.at<float>(0, i) = dst1.at<float>(0, j);//lowFilter
        src.at<float>(0, i + D / 2) = dst2.at<float>(0, j);//highFilter
    }
    return src;
}


Mat WaveTransform::waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter)
{
    assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
    assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
    Mat src = Mat_<float>(_src);

    int D = src.cols;

    Mat lowFilter = Mat_<float>(_lowFilter);
    Mat highFilter = Mat_<float>(_highFilter);


    /// 插值;
    Mat Up1 = Mat::zeros(1, D, src.type());
    Mat Up2 = Mat::zeros(1, D, src.type());


    for (int i = 0, cnt = 0; i < D / 2; i++, cnt += 2)
    {
        Up1.at<float>(0, cnt) = src.at<float>(0, i);     ///< 前一半
        Up2.at<float>(0, cnt) = src.at<float>(0, i + D / 2); ///< 后一半
    }

    /// 前一半低通，后一半高通
    Mat dst1 = Mat::zeros(1, D, src.type());
    Mat dst2 = Mat::zeros(1, D, src.type());
    filter2D(Up1, dst1, -1, lowFilter);
    filter2D(Up2, dst2, -1, highFilter);

    /// 结果相加
    dst1 = dst1 + dst2;
    return dst1;
}

void RGB2HSI(Mat src, Mat &dst) {
    Mat HSI(src.rows, src.cols, CV_32FC3);
    float r, g, b, H, S, I, num, den, theta, sum, min_RGB;
    for (int i = 0; i<src.rows; i++)
    {
        for (int j = 0; j<src.cols; j++)
        {
            b = src.at<Vec3b>(i, j)[0];
            g = src.at<Vec3b>(i, j)[1];
            r = src.at<Vec3b>(i, j)[2];

            num = 0.5 * ((r - g) + (r - b));
            den = sqrt((r - g)*(r - g) + (r - b)*(g - b));

            if (den == 0) {
                H = 0; // 分母不能为0
            }
            else {
                theta = acos(num / den);
                if (b <= g) {
                    H = theta;
                }
                else {
                    H = (2 * M_PI - theta);
                }
            }

            min_RGB = min(min(b, g), r); // min(R,G,B)
            sum = b + g + r;
            if (sum == 0)
            {
                S = 0;
            }
            else {
                S = 1 - 3 * min_RGB / sum;
            }

            I = sum / 3.0;

            HSI.at<Vec3f>(i, j)[0] = H;
            HSI.at<Vec3f>(i, j)[1] = S;
            HSI.at<Vec3f>(i, j)[2] = I;
        }
    }
    dst = HSI;
    return;
}
void HSI2RGB(Mat src, Mat &dst) {
    Mat RGB1(src.size(), CV_32FC3);
    for (int i = 0;i < src.rows;i++) {
        for (int j = 0;j < src.cols;j++) {
            float DH = src.at<Vec3f>(i, j)[0];
            float DS = src.at<Vec3f>(i, j)[1];
            float DI = src.at<Vec3f>(i, j)[2];
            //分扇区显示
            float R, G, B;
            if (DH < (2 * M_PI / 3) && DH >= 0) {
                B = DI * (1 - DS);
                R = DI * (1 + (DS * cos(DH)) / cos(M_PI / 3 - DH));
                G = (3 * DI - (R + B));
            }
            else if (DH < (4 * M_PI / 3) && DH >= (2 * M_PI / 3)) {
                DH = DH - (2 * M_PI / 3);
                R = DI * (1 - DS);
                G = DI * (1 + (DS * cos(DH)) / cos(M_PI / 3 - DH));
                B = (3 * DI - (G + R));
            }
            else {
                DH = DH - (4 * M_PI / 3);
                G = DI * (1 - DS);
                B = DI * (1 + (DS * cos(DH)) / cos(M_PI / 3 - DH));
                R = (3 * DI - (G + B));
            }
            RGB1.at<Vec3f>(i, j)[0] = B;
            RGB1.at<Vec3f>(i, j)[1] = G;
            RGB1.at<Vec3f>(i, j)[2] = R;
        }
    }
    dst = RGB1;
    return;
}
void harr_fusion(Mat src1, Mat src2,Mat &dst) {
    assert(src1.rows == src2.rows&&src1.cols == src2.cols);
    int row = src1.rows;
    int col = src1.cols;
    Mat src1_gray, src2_gray;
    normalize(src1, src1_gray, 0, 255, NORM_MINMAX);
    cvtColor(src2, src2_gray, COLOR_RGB2GRAY);
    normalize(src2_gray, src2_gray, 0, 255, NORM_MINMAX);
    src1_gray.convertTo(src1_gray, CV_32F);
    src2_gray.convertTo(src2_gray, CV_32F);
    WaveTransform m_waveTransform;
    const int level = 2;
    Mat src1_dec = m_waveTransform.WDT(src1_gray, "haar", level);
    Mat src2_dec = m_waveTransform.WDT(src2_gray, "haar", level);
    Mat dec = Mat(row,col, CV_32FC1);
    //融合规则：高频部分采用加权平均的方法，低频部分采用模值取大的方法
    int halfRow = row / (2 * level);
    int halfCol = col / (2 * level);
    for (int i = 0;i < row;i++) {
        for (int j = 0;j < col;j++) {
            if (i < halfRow&&j < halfCol) {
                dec.at<float>(i, j) = (src1_dec.at<float>(i, j) + src2_dec.at<float>(i, j)) / 2;
            }
            else {
                float p = abs(src1_dec.at<float>(i, j));
                float q = abs(src2_dec.at<float>(i, j));
                if (p > q) {
                    dec.at<float>(i, j) = src1_dec.at<float>(i, j);
                }
                else {
                    dec.at<float>(i, j) = src2_dec.at<float>(i, j);
                }

            }
        }
    }
    dst = m_waveTransform.IWDT(dec, "haar", level);
}
void fusion(Mat Visible, Mat Infrared, Mat &dst) {
    Mat HSI(Visible.size(), CV_32FC3);
    Mat Visible_I(Visible.size(), CV_32FC1);
    RGB2HSI(Visible, HSI);
    for (int i = 0;i < Visible.rows;i++) {
        for (int j = 0;j < Visible.cols;j++) {
            Visible_I.at<float>(i, j) = HSI.at<Vec3f>(i, j)[2];
        }
    }
    Mat fusion_I;
    harr_fusion(Visible_I, Infrared, fusion_I);
    Mat fusion_dst = Mat::zeros(Visible.size(), CV_32FC3);
    for (int i = 0;i < Visible.rows;i++) {
        for (int j = 0;j < Visible.cols;j++) {
            fusion_dst.at<Vec3f>(i, j)[2] = fusion_I.at<float>(i, j);
            fusion_dst.at<Vec3f>(i, j)[0] = HSI.at<Vec3f>(i, j)[0];
            fusion_dst.at<Vec3f>(i, j)[1] = HSI.at<Vec3f>(i, j)[1];
        }
    }
    HSI2RGB(fusion_dst, dst);
    dst.convertTo(dst, CV_8UC3);
    normalize(dst, dst, 0, 255, NORM_MINMAX);
}



Mat Img2;
void MainWindow::on_actionopen_file2_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.jpg *.jpeg *.bmp)"));
    Mat img = cv::imread(fileName.toStdString());
    Img2 = cv::imread(fileName.toStdString());
    cvtColor(img, img, COLOR_BGR2RGB);
    //QImage image = QImage((const unsigned char*)(image.data),image.cols,image.rows,QImage::Format_RGB888);
    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888)
            )
        );
}


void MainWindow::on_pushButton_4_clicked()
{

    Mat img = Img;
    Mat img2 = Img2;
    Mat dst;
    fusion(img,img2,dst);

    //cvtColor(dst, dst, COLOR_BGR2RGB);
    //normalize(dst, dst, 0, 255, NORM_MINMAX);
    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(dst.data, dst.cols, dst.rows, dst.step, QImage::Format_RGB888)
            )
        );

}

struct center0
{
    int x;//column
    int y;//row
    int L;
    int A;
    int B;
    int label;
};



    //input parameters:
    //imageLAB:    the source image in Lab color space
    //DisMask:       it save the shortest distance to the nearest center
    //labelMask:   it save every pixel's label
    //centers:       clustering center
    //len:         the super pixls will be initialize to len*len
    //m:           a parameter witch adjust the weights of the spacial and color space distance
    //
    //output:

int clustering(const cv::Mat &imageLAB, cv::Mat &DisMask, cv::Mat &labelMask,
                   std::vector<center0> &centers, int len, int m)
{
    if (imageLAB.empty())
    {
        std::cout << "clustering :the input image is empty!\n";
        return -1;
    }

    double *disPtr = NULL;//disMask type: 64FC1
    double *labelPtr = NULL;//labelMask type: 64FC1
    const uchar *imgPtr = NULL;//imageLAB type: 8UC3

    //disc = std::sqrt(pow(L - cL, 2)+pow(A - cA, 2)+pow(B - cB,2))
    //diss = std::sqrt(pow(x-cx,2) + pow(y-cy,2));
    //dis = sqrt(disc^2 + (diss/len)^2 * m^2)
    double dis = 0, disc = 0, diss = 0;
    //cluster center's cx, cy,cL,cA,cB;
    int cx, cy, cL, cA, cB, clabel;
    //imageLAB's  x, y, L,A,B
    int x, y, L, A, B;

    //注：这里的图像坐标以左上角为原点，水平向右为x正方向,水平向下为y正方向，与opencv保持一致
    //      从矩阵行列角度看，i表示行，j表示列，即(i,j) = (y,x)
    for (int ck = 0; ck < centers.size(); ++ck)
    {
        cx = centers[ck].x;
        cy = centers[ck].y;
        cL = centers[ck].L;
        cA = centers[ck].A;
        cB = centers[ck].B;
        clabel = centers[ck].label;

        for (int i = cy - len; i < cy + len; i++)
        {
            if (i < 0 | i >= imageLAB.rows) continue;
            //pointer point to the ith row
            imgPtr = imageLAB.ptr<uchar>(i);
            disPtr = DisMask.ptr<double>(i);
            labelPtr = labelMask.ptr<double>(i);
            for (int j = cx - len; j < cx + len; j++)
            {
                if (j < 0 | j >= imageLAB.cols) continue;
                L = *(imgPtr + j * 3);
                A = *(imgPtr + j * 3 + 1);
                B = *(imgPtr + j * 3 + 2);

                disc = std::sqrt(pow(L - cL, 2) + pow(A - cA, 2) + pow(B - cB, 2));
                diss = std::sqrt(pow(j - cx, 2) + pow(i - cy, 2));
                dis = sqrt(pow(disc, 2) + m * pow(diss, 2));

                if (dis < *(disPtr + j))
                {
                    *(disPtr + j) = dis;
                    *(labelPtr + j) = clabel;
                }//end if
            }//end for
        }
    }//end for (int ck = 0; ck < centers.size(); ++ck)


    return 0;
}



    //input parameters:
    //imageLAB:    the source image in Lab color space
    //labelMask:    it save every pixel's label
    //centers:       clustering center
    //len:         the super pixls will be initialize to len*len
    //
    //output:

int updateCenter(cv::Mat &imageLAB, cv::Mat &labelMask, std::vector<center0> &centers, int len)
{
    double *labelPtr = NULL;//labelMask type: 64FC1
    const uchar *imgPtr = NULL;//imageLAB type: 8UC3
    int cx, cy;

    for (int ck = 0; ck < centers.size(); ++ck)
    {
        double sumx = 0, sumy = 0, sumL = 0, sumA = 0, sumB = 0, sumNum = 0;
        cx = centers[ck].x;
        cy = centers[ck].y;
        for (int i = cy - len; i < cy + len; i++)
        {
            if (i < 0 | i >= imageLAB.rows) continue;
            //pointer point to the ith row
            imgPtr = imageLAB.ptr<uchar>(i);
            labelPtr = labelMask.ptr<double>(i);
            for (int j = cx - len; j < cx + len; j++)
            {
                if (j < 0 | j >= imageLAB.cols) continue;

                if (*(labelPtr + j) == centers[ck].label)
                {
                    sumL += *(imgPtr + j * 3);
                    sumA += *(imgPtr + j * 3 + 1);
                    sumB += *(imgPtr + j * 3 + 2);
                    sumx += j;
                    sumy += i;
                    sumNum += 1;
                }//end if
            }
        }
        //update center
        if (sumNum == 0) sumNum = 0.000000001;
        centers[ck].x = sumx / sumNum;
        centers[ck].y = sumy / sumNum;
        centers[ck].L = sumL / sumNum;
        centers[ck].A = sumA / sumNum;
        centers[ck].B = sumB / sumNum;

    }//end for

    return 0;
}


Mat showSLICResult(const cv::Mat &image, cv::Mat &labelMask, std::vector<center0> &centers, int len)
{
    cv::Mat dst = image.clone();
    cv::cvtColor(dst, dst, cv::COLOR_BGR2Lab);
    double *labelPtr = NULL;//labelMask type: 32FC1
    uchar *imgPtr = NULL;//image type: 8UC3

    int cx, cy;
    double sumx = 0, sumy = 0, sumL = 0, sumA = 0, sumB = 0, sumNum = 0.00000001;
    for (int ck = 0; ck < centers.size(); ++ck)
    {
        cx = centers[ck].x;
        cy = centers[ck].y;

        for (int i = cy - len; i < cy + len; i++)
        {
            if (i < 0 | i >= image.rows) continue;
            //pointer point to the ith row
            imgPtr = dst.ptr<uchar>(i);
            labelPtr = labelMask.ptr<double>(i);
            for (int j = cx - len; j < cx + len; j++)
            {
                if (j < 0 | j >= image.cols) continue;

                if (*(labelPtr + j) == centers[ck].label)
                {
                    *(imgPtr + j * 3) = centers[ck].L;
                    *(imgPtr + j * 3 + 1) = centers[ck].A;
                    *(imgPtr + j * 3 + 2) = centers[ck].B;
                }//end if
            }
        }
    }//end for

    cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);

//    cv::namedWindow("showSLIC", 0);
//    cv::imshow("showSLIC", dst);
//    cv::waitKey(1);

    return dst;
}


Mat showSLICResult2(const cv::Mat &image, cv::Mat &labelMask, std::vector<center0> &centers, int len)
{
    cv::Mat dst = image.clone();
    //cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);
    double *labelPtr = NULL;//labelMask type: 32FC1
    double *labelPtr_nextRow = NULL;//labelMask type: 32FC1
    uchar *imgPtr = NULL;//image type: 8UC3

    for (int i = 0; i < labelMask.rows - 1; i++)
    {
        labelPtr = labelMask.ptr<double>(i);
        imgPtr = dst.ptr<uchar>(i);
        for (int j = 0; j < labelMask.cols - 1; j++)
        {
            //if left pixel's label is different from the right's
            if (*(labelPtr + j) != *(labelPtr + j + 1))
            {
                *(imgPtr + 3 * j) = 0;
                *(imgPtr + 3 * j + 1) = 0;
                *(imgPtr + 3 * j + 2) = 0;
            }

            //if the upper pixel's label is different from the bottom's
            labelPtr_nextRow = labelMask.ptr<double>(i + 1);
            if (*(labelPtr_nextRow + j) != *(labelPtr + j))
            {
                *(imgPtr + 3 * j) = 0;
                *(imgPtr + 3 * j + 1) = 0;
                *(imgPtr + 3 * j + 2) = 0;
            }
        }
    }

    //show center
    for (int ck = 0; ck < centers.size(); ck++)
    {
        imgPtr = dst.ptr<uchar>(centers[ck].y);
        *(imgPtr + centers[ck].x * 3) = 100;
        *(imgPtr + centers[ck].x * 3 + 1) = 100;
        *(imgPtr + centers[ck].x * 3 + 1) = 10;
    }

//    cv::namedWindow("showSLIC2", 0);
//    cv::imshow("showSLIC2", dst);
//    cv::waitKey(1);
    return dst;
}


int initilizeCenters(cv::Mat &imageLAB, std::vector<center0> &centers, int len)
{
    if (imageLAB.empty())
    {
        std::cout << "In itilizeCenters:     image is empty!\n";
        return -1;
    }

    uchar *ptr = NULL;
    center0 cent;
    int num = 0;
    for (int i = 0; i < imageLAB.rows; i += len)
    {
        cent.y = i + len / 2;
        if (cent.y >= imageLAB.rows) continue;
        ptr = imageLAB.ptr<uchar>(cent.y);
        for (int j = 0; j < imageLAB.cols; j += len)
        {
            cent.x = j + len / 2;
            if ((cent.x >= imageLAB.cols)) continue;
            cent.L = *(ptr + cent.x * 3);
            cent.A = *(ptr + cent.x * 3 + 1);
            cent.B = *(ptr + cent.x * 3 + 2);
            cent.label = ++num;
            centers.push_back(cent);
        }
    }
    return 0;
}


//if the center locates in the edges, fitune it's location.
int fituneCenter(cv::Mat &imageLAB, cv::Mat &sobelGradient, std::vector<center0> &centers)
{
    if (sobelGradient.empty()) return -1;

    center0 cent;
    double *sobPtr = sobelGradient.ptr<double>(0);
    uchar *imgPtr = imageLAB.ptr<uchar>(0);
    int w = sobelGradient.cols;
    for (int ck = 0; ck < centers.size(); ck++)
    {
        cent = centers[ck];
        if (cent.x - 1 < 0 || cent.x + 1 >= sobelGradient.cols || cent.y - 1 < 0 || cent.y + 1 >= sobelGradient.rows)
        {
            continue;
        }//end if
        double minGradient = 9999999;
        int tempx = 0, tempy = 0;
        for (int m = -1; m < 2; m++)
        {
            sobPtr = sobelGradient.ptr<double>(cent.y + m);
            for (int n = -1; n < 2; n++)
            {
                double gradient = pow(*(sobPtr + (cent.x + n) * 3), 2)
                                  + pow(*(sobPtr + (cent.x + n) * 3 + 1), 2)
                                  + pow(*(sobPtr + (cent.x + n) * 3 + 2), 2);
                if (gradient < minGradient)
                {
                    minGradient = gradient;
                    tempy = m;//row
                    tempx = n;//column
                }//end if
            }
        }
        cent.x += tempx;
        cent.y += tempy;
        imgPtr = imageLAB.ptr<uchar>(cent.y);
        centers[ck].x = cent.x;
        centers[ck].y = cent.y;
        centers[ck].L = *(imgPtr + cent.x * 3);
        centers[ck].A = *(imgPtr + cent.x * 3 + 1);
        centers[ck].B = *(imgPtr + cent.x * 3 + 2);

    }//end for
    return 0;
}



    //input parameters:
    //image:    the source image in RGB color space
    //resultLabel:     it save every pixel's label
    //len:         the super pixls will be initialize to len*len
    //m:           a parameter witch adjust the weights of diss
    //output:

int SLIC(cv::Mat &image, cv::Mat &resultLabel, std::vector<center0> &centers, int len, int m)
{
    if (image.empty())
    {
        std::cout << "in SLIC the input image is empty!\n";
        return -1;

    }

    int MAXDIS = 999999;
    int height, width;
    height = image.rows;
    width = image.cols;

    //convert color
    cv::Mat imageLAB;
    cv::cvtColor(image, imageLAB, cv::COLOR_BGR2Lab);

    //get sobel gradient map
    cv::Mat sobelImagex, sobelImagey, sobelGradient;
    cv::Sobel(imageLAB, sobelImagex, CV_64F, 0, 1, 3);
    cv::Sobel(imageLAB, sobelImagey, CV_64F, 1, 0, 3);
    cv::addWeighted(sobelImagex, 0.5, sobelImagey, 0.5, 0, sobelGradient);//sobel output image type is CV_64F

    //initiate
    //std::vector<center> centers;
    //disMask save the distance of the pixels to center;
    cv::Mat disMask ;
    //labelMask save the label of the pixels
    cv::Mat labelMask = cv::Mat::zeros(cv::Size(width, height), CV_64FC1);

    //initialize centers,  get centers
    initilizeCenters(imageLAB, centers, len);
    //if the center locates in the edges, fitune it's location
    fituneCenter(imageLAB, sobelGradient, centers);

    //update cluster 10 times
    for (int time = 0; time < 10; time++)
    {
        //clustering
        disMask = cv::Mat(height, width, CV_64FC1, cv::Scalar(MAXDIS));
        clustering(imageLAB, disMask, labelMask, centers, len, m);
        //update
        updateCenter(imageLAB, labelMask, centers, len);
        //fituneCenter(imageLAB, sobelGradient, centers);
    }

    resultLabel = labelMask;

    return 0;
}


void MainWindow::on_pushButton_5_clicked()
{
    Mat img = Img;
    cvtColor(img, img, COLOR_BGR2RGB);


    Mat labelMask;//save every pixel's label
    Mat img2;//save the shortest distance to the nearest centers
    Mat img3;
    std::vector<center0> centers;//clustering centers

    int len = 10;//the scale of the superpixel ,len*len
    int m = 10;//a parameter witch adjust the weights of spacial distance and the color space distance
    SLIC(img, labelMask, centers, len, m);


    img2 = showSLICResult(img, labelMask, centers, len);
    img3 = showSLICResult2(img, labelMask, centers, len);


    ui->label->setPixmap(
        QPixmap::fromImage(
            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888)
            )
        );

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(img2.data, img2.cols, img2.rows, img2.step, QImage::Format_RGB888)
            )
        );

    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(img3.data, img3.cols, img3.rows, img3.step, QImage::Format_RGB888)
            )
        );



}

