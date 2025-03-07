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
Mat Img3;
void MainWindow::on_actionopen_file_triggered()
{

    int i, j;
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.jpg *.jpeg *.bmp)"));
    Mat img = cv::imread(fileName.toStdString());
    //Img = cv::imread(fileName.toStdString());
    cvtColor(img, img, COLOR_BGR2RGB);
    Img = img;


    //QImage image = QImage((const unsigned char*)(image.data),image.cols,image.rows,QImage::Format_RGB888);
    ui->label->setPixmap(
        QPixmap::fromImage(
            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888)
            )
        );
    QImage img1(img.cols, img.rows, QImage::Format_RGB888);
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);
    QImage img3(img.cols, img.rows, QImage::Format_RGB888);
    QImage img4(img.cols, img.rows, QImage::Format_RGB888);

    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {
            // Set the pixel value of the QImage
            //QImage img3 = img;
            //QColor c = img.pixel(i,j);
            int b = img.at<Vec3b>(i,j)[0];
            int g = img.at<Vec3b>(i,j)[1];
            int r = img.at<Vec3b>(i,j)[2];
            img1.setPixel(j,i,qRgb(r,r,r));
            img2.setPixel(j,i,qRgb(g,g,g));
            img3.setPixel(j,i,qRgb(b,b,b));
            img4.setPixel(j,i,qRgb(b,g,r));
        }
    }

    ui->label_2->setPixmap(QPixmap::fromImage(img1));
    ui->label_3->setPixmap(QPixmap::fromImage(img2));
    ui->label_4->setPixmap(QPixmap::fromImage(img3));
    ui->label_5->setPixmap(QPixmap::fromImage(img4));
}


void MainWindow::on_pushButton_clicked()
{
    Mat img = Img;
    //cvtColor(img, img, COLOR_BGR2RGB);

    Mat img4 = img;//(img.cols, img.rows, QImage::Format_RGB888);

    //cvtColor(img, img4, COLOR_RGB2HLS);

    vector<Mat>channel4s;


    //cvtColor(img, img4, COLOR_BGR2RGB);
    split(img4, channel4s);
    //Mat newimg4;
    //cvtColor(img4, newimg4, COLOR_HLS2RGB);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[0].data, channel4s[0].cols, channel4s[0].rows, channel4s[0].step, QImage::Format_Alpha8)
            )
        );
    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[1].data, channel4s[1].cols, channel4s[1].rows, channel4s[1].step, QImage::Format_Alpha8)
            )
        );
    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[2].data, channel4s[2].cols, channel4s[2].rows, channel4s[2].step, QImage::Format_Alpha8)
            )
        );
    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(img4.data, img4.cols, img4.rows, img4.step, QImage::Format_RGB888)
            )
        );
}


void MainWindow::on_pushButton_2_clicked()
{

    Mat img = Img;
    //cvtColor(img, img, COLOR_BGR2RGB);

    Mat img4;//(img.cols, img.rows, QImage::Format_RGB888);

    cvtColor(img, img4, COLOR_RGB2HLS);

    vector<Mat>channel4s;

    split(img4, channel4s);

    Mat newimg4;
    cvtColor(img4, newimg4, COLOR_HLS2RGB);
    //cvtColor(newimg4, newimg4, COLOR_BGR2RGB);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[0].data, channel4s[0].cols, channel4s[0].rows, channel4s[0].step, QImage::Format_Alpha8)
            )
        );
    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[1].data, channel4s[1].cols, channel4s[1].rows, channel4s[1].step, QImage::Format_Alpha8)
            )
        );
    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[2].data, channel4s[2].cols, channel4s[2].rows, channel4s[2].step, QImage::Format_Alpha8)
            )
        );
    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(newimg4.data, newimg4.cols, newimg4.rows, newimg4.step, QImage::Format_RGB888)
            )
        );

}


void MainWindow::on_pushButton_3_clicked()
{
    Mat img = Img;
    //cvtColor(img, img, COLOR_BGR2RGB);

    Mat img4;//(img.cols, img.rows, QImage::Format_RGB888);

    img4.create(img.rows,img.cols,CV_8UC4);
    rgb2cmyk(img,img4);
    //cvtColor(img, img4, COLOR_BGR2);

    vector<Mat>channel4s;

    split(img4, channel4s);

    //Mat newimg4;
    //cvtColor(img4, newimg4, COLOR_HLS2RGB);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[0].data, channel4s[0].cols, channel4s[0].rows, channel4s[0].step, QImage::Format_Alpha8)
            )
        );
    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[1].data, channel4s[1].cols, channel4s[1].rows, channel4s[1].step, QImage::Format_Alpha8)
            )
        );
    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[2].data, channel4s[2].cols, channel4s[2].rows, channel4s[2].step, QImage::Format_Alpha8)
            )
        );
    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888)
            )
        );
}


void MainWindow::on_pushButton_4_clicked()
{
    Mat img = Img;
    //cvtColor(img, img, COLOR_BGR2RGB);

    Mat img4;//(img.cols, img.rows, QImage::Format_RGB888);

    cvtColor(img, img4, COLOR_RGB2XYZ);

    vector<Mat>channel4s;

    split(img4, channel4s);

    Mat newimg4;
    cvtColor(img4, newimg4, COLOR_XYZ2RGB);
    //cvtColor(newimg4, newimg4, COLOR_BGR2RGB);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[0].data, channel4s[0].cols, channel4s[0].rows, channel4s[0].step, QImage::Format_Alpha8)
            )
        );
    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[1].data, channel4s[1].cols, channel4s[1].rows, channel4s[1].step, QImage::Format_Alpha8)
            )
        );
    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[2].data, channel4s[2].cols, channel4s[2].rows, channel4s[2].step, QImage::Format_Alpha8)
            )
        );
    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(newimg4.data, newimg4.cols, newimg4.rows, newimg4.step, QImage::Format_RGB888)
            )
        );
}


void MainWindow::on_pushButton_5_clicked()
{
    Mat img = Img;
    //cvtColor(img, img, COLOR_BGR2RGB);

    Mat img4;//(img.cols, img.rows, QImage::Format_RGB888);

    cvtColor(img, img4, COLOR_RGB2Lab);

    vector<Mat>channel4s;

    split(img4, channel4s);

    Mat newimg4;
    cvtColor(img4, newimg4, COLOR_Lab2RGB);
    //cvtColor(newimg4, newimg4, COLOR_BGR2RGB);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[0].data, channel4s[0].cols, channel4s[0].rows, channel4s[0].step, QImage::Format_Alpha8)
            )
        );
    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[1].data, channel4s[1].cols, channel4s[1].rows, channel4s[1].step, QImage::Format_Alpha8)
            )
        );
    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[2].data, channel4s[2].cols, channel4s[2].rows, channel4s[2].step, QImage::Format_Alpha8)
            )
        );
    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(newimg4.data, newimg4.cols, newimg4.rows, newimg4.step, QImage::Format_RGB888)
            )
        );
}


void MainWindow::on_pushButton_6_clicked()
{
    Mat img = Img;
    //cvtColor(img, img, COLOR_BGR2RGB);

    Mat img4;//(img.cols, img.rows, QImage::Format_RGB888);

    cvtColor(img, img4, COLOR_RGB2YUV);

    vector<Mat>channel4s;

    split(img4, channel4s);

    Mat newimg4;
    cvtColor(img4, newimg4, COLOR_YUV2RGB);
    //cvtColor(newimg4, newimg4, COLOR_BGR2RGB);

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[0].data, channel4s[0].cols, channel4s[0].rows, channel4s[0].step, QImage::Format_Alpha8)
            )
        );
    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[1].data, channel4s[1].cols, channel4s[1].rows, channel4s[1].step, QImage::Format_Alpha8)
            )
        );
    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(channel4s[2].data, channel4s[2].cols, channel4s[2].rows, channel4s[2].step, QImage::Format_Alpha8)
            )
        );
    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(newimg4.data, newimg4.cols, newimg4.rows, newimg4.step, QImage::Format_RGB888)
            )
        );
}

Mat Img2;
void MainWindow::on_actionpart_2_grayscale_file_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.jpg *.jpeg *.bmp)"));
    Mat img = cv::imread(fileName.toStdString(),IMREAD_GRAYSCALE);
    Img2 = cv::imread(fileName.toStdString(),IMREAD_GRAYSCALE);
    ui->label->setPixmap(
        QPixmap::fromImage(
            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_Grayscale8)
            )
        );

}

//string text = "COLORMAP_HSV";
void MainWindow::on_pushButton_7_clicked()
{
    Mat img = Img3;
    //定义随机颜色
    cv::Scalar colorTab[] = {
        Scalar(0,0,255),
        Scalar(0,255,0),
        Scalar(255,0,0),
        Scalar(0,255,255),
        Scalar(255,0,255)
    };
    //获取原图属性，宽高及维度
    int width = img.cols;
    int height = img.rows;
    int dims = img.channels();

    //初始化采样数量
    int sampleCount= width*height;
    int clusterCount = 2;//四分类
    Mat points(sampleCount,dims,CV_32F,Scalar(10));
    Mat labels;
    Mat centers(clusterCount,1,points.type());


    int index = 0;
    for(int row = 0;row<height;row++){
        for(int col = 0;col<width;col++){
            index = row*width+col;
            Vec3b bgr = img.at<Vec3b>(row,col);
            points.at<float>(index,0) = static_cast<int>(bgr[0]);
            points.at<float>(index,1) = static_cast<int>(bgr[1]);
            points.at<float>(index,2) = static_cast<int>(bgr[2]);
        }
    }

    TermCriteria criteria = TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,10,0.1);
    kmeans(points,clusterCount,labels,criteria,3,KMEANS_PP_CENTERS,centers);

    Mat result = Mat::zeros(img.size(),img.type());
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            index  = row*width+col;
            int label = labels.at<int>(index,0);
            result.at<Vec3b>(row,col)[0] = colorTab[label][0];
            result.at<Vec3b>(row,col)[1] = colorTab[label][1];
            result.at<Vec3b>(row,col)[2] = colorTab[label][2];
        }
    }

    int clusterCount3 = 3;//四分类
    //Mat points(sampleCount,dims,CV_32F,Scalar(10));
    Mat labels3;
    Mat centers3(clusterCount3,1,points.type());

    //TermCriteria criteria = TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,10,0.1);
    kmeans(points,clusterCount3,labels3,criteria,3,KMEANS_PP_CENTERS,centers3);

    Mat result3 = Mat::zeros(img.size(),img.type());
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            index  = row*width+col;
            int label = labels3.at<int>(index,0);
            result3.at<Vec3b>(row,col)[0] = colorTab[label][0];
            result3.at<Vec3b>(row,col)[1] = colorTab[label][1];
            result3.at<Vec3b>(row,col)[2] = colorTab[label][2];
        }
    }

    int clusterCount4 = 4;//四分类
    //Mat points(sampleCount,dims,CV_32F,Scalar(10));
    Mat labels4;
    Mat centers4(clusterCount4,1,points.type());

    //TermCriteria criteria = TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,10,0.1);
    kmeans(points,clusterCount4,labels4,criteria,3,KMEANS_PP_CENTERS,centers4);

    Mat result4 = Mat::zeros(img.size(),img.type());
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            index  = row*width+col;
            int label = labels4.at<int>(index,0);
            result4.at<Vec3b>(row,col)[0] = colorTab[label][0];
            result4.at<Vec3b>(row,col)[1] = colorTab[label][1];
            result4.at<Vec3b>(row,col)[2] = colorTab[label][2];
        }
    }

    int clusterCount5 = 5;//四分类
    //Mat points(sampleCount,dims,CV_32F,Scalar(10));
    Mat labels5;
    Mat centers5(clusterCount5,1,points.type());

    //TermCriteria criteria = TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,10,0.1);
    kmeans(points,clusterCount5,labels5,criteria,3,KMEANS_PP_CENTERS,centers5);

    Mat result5 = Mat::zeros(img.size(),img.type());
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            index  = row*width+col;
            int label = labels5.at<int>(index,0);
            result5.at<Vec3b>(row,col)[0] = colorTab[label][0];
            result5.at<Vec3b>(row,col)[1] = colorTab[label][1];
            result5.at<Vec3b>(row,col)[2] = colorTab[label][2];
        }
    }

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(result.data, result.cols, result.rows, result.step, QImage::Format_RGB888)
            )
        );

    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(result3.data, result3.cols, result3.rows, result3.step, QImage::Format_RGB888)
            )
        );

    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(result4.data, result4.cols, result4.rows, result4.step, QImage::Format_RGB888)
            )
        );

    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(result5.data, result5.cols, result5.rows, result5.step, QImage::Format_RGB888)
            )
        );



}


void MainWindow::on_pushButton_8_clicked()
{
    Mat img = Img3;
    Mat img2;
    Mat img3;

    cvtColor(img, img2, COLOR_RGB2HLS);
    cvtColor(img, img3, COLOR_RGB2Lab);

    //定义随机颜色
    cv::Scalar colorTab[] = {
        Scalar(0,0,255),
        Scalar(0,255,0),
        Scalar(255,0,0),
        Scalar(0,255,255),
        Scalar(255,0,255)
    };
    //获取原图属性，宽高及维度
    int width = img.cols;
    int height = img.rows;
    int dims = img.channels();

    int dims2 = img2.channels();

    int dims3 = img3.channels();

    //初始化采样数量
    int sampleCount= width*height;
    int clusterCount = 2;//四分类
    Mat points(sampleCount,dims,CV_32F,Scalar(10));
    Mat labels;
    Mat centers(clusterCount,1,points.type());

    Mat points2(sampleCount,dims2,CV_32F,Scalar(10));
    Mat labels2;
    Mat centers2(clusterCount,1,points2.type());

    Mat points3(sampleCount,dims3,CV_32F,Scalar(10));
    Mat labels3;
    Mat centers3(clusterCount,1,points3.type());



    int index = 0;
    for(int row = 0;row<height;row++){
        for(int col = 0;col<width;col++){
            index = row*width+col;
            Vec3b bgr = img.at<Vec3b>(row,col);
            points.at<float>(index,0) = static_cast<int>(bgr[0]);
            points.at<float>(index,1) = static_cast<int>(bgr[1]);
            points.at<float>(index,2) = static_cast<int>(bgr[2]);
        }
    }

    for(int row = 0;row<height;row++){
        for(int col = 0;col<width;col++){
            index = row*width+col;
            Vec3b bgr = img2.at<Vec3b>(row,col);
            points2.at<float>(index,0) = static_cast<int>(bgr[0]);
            points2.at<float>(index,1) = static_cast<int>(bgr[1]);
            points2.at<float>(index,2) = static_cast<int>(bgr[2]);
        }
    }

    for(int row = 0;row<height;row++){
        for(int col = 0;col<width;col++){
            index = row*width+col;
            Vec3b bgr = img3.at<Vec3b>(row,col);
            points3.at<float>(index,0) = static_cast<int>(bgr[0]);
            points3.at<float>(index,1) = static_cast<int>(bgr[1]);
            points3.at<float>(index,2) = static_cast<int>(bgr[2]);
        }
    }


    TermCriteria criteria = TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,10,0.1);
    kmeans(points,clusterCount,labels,criteria,3,KMEANS_PP_CENTERS,centers);

    kmeans(points2,clusterCount,labels2,criteria,3,KMEANS_PP_CENTERS,centers2);

    kmeans(points3,clusterCount,labels3,criteria,3,KMEANS_PP_CENTERS,centers3);

    Mat result = Mat::zeros(img.size(),img.type());
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            index  = row*width+col;
            int label = labels.at<int>(index,0);
            result.at<Vec3b>(row,col)[0] = colorTab[label][0];
            result.at<Vec3b>(row,col)[1] = colorTab[label][1];
            result.at<Vec3b>(row,col)[2] = colorTab[label][2];
        }
    }
    Mat result2 = Mat::zeros(img2.size(),img2.type());
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            index  = row*width+col;
            int label = labels2.at<int>(index,0);
            result2.at<Vec3b>(row,col)[0] = colorTab[label][0];
            result2.at<Vec3b>(row,col)[1] = colorTab[label][1];
            result2.at<Vec3b>(row,col)[2] = colorTab[label][2];
        }
    }
    Mat result3 = Mat::zeros(img3.size(),img3.type());
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            index  = row*width+col;
            int label = labels3.at<int>(index,0);
            result3.at<Vec3b>(row,col)[0] = colorTab[label][0];
            result3.at<Vec3b>(row,col)[1] = colorTab[label][1];
            result3.at<Vec3b>(row,col)[2] = colorTab[label][2];
        }
    }



    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(result.data, result.cols, result.rows, result.step, QImage::Format_RGB888)
            )
        );

    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(result2.data, result2.cols, result2.rows, result2.step, QImage::Format_RGB888)
            )
        );

    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(result3.data, result3.cols, result3.rows, result3.step, QImage::Format_RGB888)
            )
        );


//    Mat lut(1, 256, CV_8U);
//    for (int i = 0; i < 256; i++) {
//        lut.at<uchar>(i) = 255 - i;
//    }
//    Mat img2;
//    LUT(img, lut, img2);

//    ui->label_2->setPixmap(
//        QPixmap::fromImage(
//            QImage(img2.data, img2.cols, img2.rows, img2.step, QImage::Format_Grayscale8)
//            )
//        );


//    Mat lut(30, 256, CV_8U);

//    for (int i = 0; i < 256*30; i++)
//    {
//        lut.at<uchar>(i) = i % 256;
//    }
//    Mat img3;
//    applyColorMap(lut, img3, COLORMAP_HSV);


//    ui->label_3->setPixmap(
//        QPixmap::fromImage(
//            QImage(img3.data, img3.cols, img3.rows, img3.step, QImage::Format_RGB888)
//            )
//        );


}





void MainWindow::on_pushButton_9_clicked()
{
    Mat img0;
    ui->label->setPixmap(
        QPixmap::fromImage(
            QImage(img0.data, img0.cols, img0.rows, img0.step, QImage::Format_RGB888)
            )
        );

    ui->label_2->setPixmap(
        QPixmap::fromImage(
            QImage(img0.data, img0.cols, img0.rows, img0.step, QImage::Format_RGB888)
            )
        );

    ui->label_3->setPixmap(
        QPixmap::fromImage(
            QImage(img0.data, img0.cols, img0.rows, img0.step, QImage::Format_RGB888)
            )
        );

    ui->label_4->setPixmap(
        QPixmap::fromImage(
            QImage(img0.data, img0.cols, img0.rows, img0.step, QImage::Format_RGB888)
            )
        );

    ui->label_5->setPixmap(
        QPixmap::fromImage(
            QImage(img0.data, img0.cols, img0.rows, img0.step, QImage::Format_RGB888)
            )
        );


//    int h = ui->label->height();
//    int w = ui->label->width();
//    QPixmap pix(w, h);
//    QPainter paint(&pix);
//    pix.fill( Qt::white );
//    paint.setPen(QColor(0, 0, 0, 255));
//    paint.drawRect(QRect(80,120,200,100));
//    ui->label->setPixmap(pix);
}


//char arg;
void MainWindow::on_comboBox_currentTextChanged(const QString &arg1)
{
    Mat img = Img2;
    Mat img2;

    Mat lut(30, 256, CV_8U);

    for (int i = 0; i < 256*30; i++)
    {
        lut.at<uchar>(i) = i % 256;
    }
    Mat img3;

    if(arg1 == "COLORMAP_HSV"){
        applyColorMap(img, img2, COLORMAP_HSV);
        applyColorMap(lut, img3, COLORMAP_HSV);}
    else if(arg1 == "COLORMAP_JET"){
        applyColorMap(img, img2, COLORMAP_JET);
        applyColorMap(lut, img3, COLORMAP_JET);}
    else if(arg1 == "COLORMAP_OCEAN"){
        applyColorMap(img, img2, COLORMAP_OCEAN);
        applyColorMap(lut, img3, COLORMAP_OCEAN);}
    else if(arg1 == "COLORMAP_HOT"){
        applyColorMap(img, img2, COLORMAP_HOT);
        applyColorMap(lut, img3, COLORMAP_HOT);}
    else if(arg1 == "COLORMAP_AUTUMN"){
        applyColorMap(img, img2, COLORMAP_AUTUMN);
        applyColorMap(lut, img3, COLORMAP_AUTUMN);}
    else if(arg1 == "COLORMAP_BONE"){
        applyColorMap(img, img2, COLORMAP_BONE);
        applyColorMap(lut, img3, COLORMAP_BONE);}
    else if(arg1 == "COLORMAP_WINTER"){
        applyColorMap(img, img2, COLORMAP_WINTER);
        applyColorMap(lut, img3, COLORMAP_WINTER);}
    else if(arg1 == "COLORMAP_SUMMER"){
        applyColorMap(img, img2, COLORMAP_SUMMER);
        applyColorMap(lut, img3, COLORMAP_SUMMER);}
    else if(arg1 == "COLORMAP_COOL"){
        applyColorMap(img, img2, COLORMAP_COOL);
        applyColorMap(lut, img3, COLORMAP_COOL);}
    else if(arg1 == "COLORMAP_PINK"){
        applyColorMap(img, img2, COLORMAP_PINK);
        applyColorMap(lut, img3, COLORMAP_PINK);}
    else if(arg1 == "COLORMAP_RAINBOW"){
        applyColorMap(img, img2, COLORMAP_RAINBOW);
        applyColorMap(lut, img3, COLORMAP_RAINBOW);}
    else if(arg1 == "COLORMAP_PARULA"){
        applyColorMap(img, img2, COLORMAP_PARULA);
        applyColorMap(lut, img3, COLORMAP_PARULA);}
    else if(arg1 == "COLORMAP_SPRING"){
        applyColorMap(img, img2, COLORMAP_SPRING);
        applyColorMap(lut, img3, COLORMAP_SPRING);}
    else if(arg1 == "COLORMAP_MAGMA"){
        applyColorMap(img, img2, COLORMAP_MAGMA);
        applyColorMap(lut, img3, COLORMAP_MAGMA);}
    else if(arg1 == "COLORMAP_INFERNO"){
        applyColorMap(img, img2, COLORMAP_INFERNO);
        applyColorMap(lut, img3, COLORMAP_INFERNO);}
    else if(arg1 == "COLORMAP_PLASMA"){
        applyColorMap(img, img2, COLORMAP_PLASMA);
        applyColorMap(lut, img3, COLORMAP_PLASMA);}
    else if(arg1 == "COLORMAP_VIRIDIS"){
        applyColorMap(img, img2, COLORMAP_VIRIDIS);
        applyColorMap(lut, img3, COLORMAP_VIRIDIS);}
    else if(arg1 == "COLORMAP_CIVIDIS"){
        applyColorMap(img, img2, COLORMAP_CIVIDIS);
        applyColorMap(lut, img3, COLORMAP_CIVIDIS);}
    else if(arg1 == "COLORMAP_TWILIGHT"){
        applyColorMap(img, img2, COLORMAP_TWILIGHT);
        applyColorMap(lut, img3, COLORMAP_TWILIGHT);}
    else if(arg1 == "COLORMAP_TWILIGHT_SHIFTED"){
        applyColorMap(img, img2, COLORMAP_TWILIGHT_SHIFTED);
        applyColorMap(lut, img3, COLORMAP_TWILIGHT_SHIFTED);}


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



void MainWindow::on_actionpart_3_color_file_triggered()
{
    //int i, j;
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.jpg *.jpeg *.bmp)"));
    Mat img = cv::imread(fileName.toStdString());
    //Img = cv::imread(fileName.toStdString());
    cvtColor(img, img, COLOR_BGR2RGB);
    Img3 = img;


    //QImage image = QImage((const unsigned char*)(image.data),image.cols,image.rows,QImage::Format_RGB888);
    ui->label->setPixmap(
        QPixmap::fromImage(
            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888)
            )
        );
}

