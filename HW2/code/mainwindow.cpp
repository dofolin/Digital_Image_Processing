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
    //ui->label->setScaledContents(true);
    //ui->label->show();

}




















void MainWindow::on_pushButton_clicked()
{
    int i,j;
    Mat img = Img;
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);
    for( i=0; i<256; i++ ) histogram[i] = 0;
    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {
            // Set the pixel value of the QImage
            //QImage img3 = img;
            //QColor c = img.pixel(i,j);
            int m = (img.at<Vec3b>(i,j)[0] + img.at<Vec3b>(i,j)[1] + img.at<Vec3b>(i,j)[2])/3;
            img2.setPixel(j,i,qRgb(m,m,m));
            histogram[ m ]++;
        }
    }
    ui->label_2->setPixmap(QPixmap::fromImage(img2));
    //(img2, img2, COLOR_BGR2RGB);
    //QImage image = QImage((const unsigned char*)(image.data),image.cols,image.rows,QImage::Format_RGB888);
//    ui->label_2->setPixmap(
//        QPixmap::fromImage(
//            QImage(img2.data, img2.cols, img2.rows, img2.step, QImage::Format_RGB888)
//            )
//        );
    // Calculate the histogram of the 64x64 image;
    //unsigned char image[img.rows][img.cols];
//    for( i=0; i<256; i++ ) histogram[i] = 0;	/* Initialize the array */
//    for(i=0; i<img.rows; i++)
//    {
//        for(j=0; j<img.cols; j++)
//        {
//            histogram[ img2[i][j] ]++;
//        }
//      }


    // Use QChart to display the image histogram

    QBarSet *set0 = new QBarSet("Histogram");

    for(i=0;i<256;i++)
        *set0 << histogram[i];

    QBarSeries *series = new QBarSeries();
    series->append(set0);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Histogram");
    chart->setAnimationOptions(QChart::SeriesAnimations);
    chart->legend()->setVisible(false);
    chart->legend()->setAlignment(Qt::AlignBottom);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    while(!ui->horizontalLayout->isEmpty())
    {
        // Clear the horizontal layout content if there is any
        ui->horizontalLayout->removeItem(ui->horizontalLayout->itemAt(0));
    }
    ui->horizontalLayout->addWidget(chartView);
}


void MainWindow::on_pushButton_2_clicked()
{
    int i,j;
    Mat img = Img;
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);
    for( i=0; i<256; i++ ) histogram[i] = 0;
    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {

            int m = img.at<Vec3b>(i,j)[0]*0.114 + img.at<Vec3b>(i,j)[1]*0.587 + img.at<Vec3b>(i,j)[2]*0.299;
            img2.setPixel(j,i,qRgb(m,m,m));
            histogram[ m ]++;

        }
    }
    ui->label_2->setPixmap(QPixmap::fromImage(img2));

    QBarSet *set0 = new QBarSet("Histogram");

    for(i=0;i<256;i++)
        *set0 << histogram[i];

    QBarSeries *series = new QBarSeries();
    series->append(set0);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Histogram");
    chart->setAnimationOptions(QChart::SeriesAnimations);
    chart->legend()->setVisible(false);
    chart->legend()->setAlignment(Qt::AlignBottom);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    while(!ui->horizontalLayout->isEmpty())
    {
        // Clear the horizontal layout content if there is any
        ui->horizontalLayout->removeItem(ui->horizontalLayout->itemAt(0));
    }
    ui->horizontalLayout->addWidget(chartView);
}


void MainWindow::on_pushButton_3_clicked()
{
    int i,j;
    Mat img = Img;
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);
    for( i=0; i<256; i++ ) histogram[i] = 0;
    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {
            //(B+G+R)/3-(B*0.114+G*0.587+R*0.299)
            int m = img.at<Vec3b>(i,j)[0]*0.22 + img.at<Vec3b>(i,j)[1]*(-0.254) + img.at<Vec3b>(i,j)[2]*0.034;
            int n = qBound(0, m, 255);
            img2.setPixel(j,i,qRgb(n,n,n));
            histogram[ n ]++;

        }
    }
    ui->label_2->setPixmap(QPixmap::fromImage(img2));

    QBarSet *set0 = new QBarSet("Histogram");

    for(i=0;i<256;i++)
        *set0 << histogram[i];

    QBarSeries *series = new QBarSeries();
    series->append(set0);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Histogram");
    chart->setAnimationOptions(QChart::SeriesAnimations);
    chart->legend()->setVisible(false);
    chart->legend()->setAlignment(Qt::AlignBottom);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    while(!ui->horizontalLayout->isEmpty())
    {
        // Clear the horizontal layout content if there is any
        ui->horizontalLayout->removeItem(ui->horizontalLayout->itemAt(0));
    }
    ui->horizontalLayout->addWidget(chartView);
}


void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    ui->label_3->setText(QString::number(position));
    int i,j;
    Mat img = Img;
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);
    for( i=0; i<256; i++ ) histogram[i] = 0;
    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {

            int m = img.at<Vec3b>(i,j)[0]*0.114 + img.at<Vec3b>(i,j)[1]*0.587 + img.at<Vec3b>(i,j)[2]*0.299;
            if (m >= position)
                m = 255;
            else
                m = 0;

            img2.setPixel(j,i,qRgb(m,m,m));
            histogram[ m ]++;

        }
    }
    ui->label_2->setPixmap(QPixmap::fromImage(img2));

    QBarSet *set0 = new QBarSet("Histogram");

    for(i=0;i<256;i++)
        *set0 << histogram[i];

    QBarSeries *series = new QBarSeries();
    series->append(set0);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Histogram");
    chart->setAnimationOptions(QChart::SeriesAnimations);
    chart->legend()->setVisible(false);
    chart->legend()->setAlignment(Qt::AlignBottom);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);


    while(!ui->horizontalLayout->isEmpty())
    {
        // Clear the horizontal layout content if there is any
        ui->horizontalLayout->removeItem(ui->horizontalLayout->itemAt(0));
    }
    ui->horizontalLayout->addWidget(chartView);
}


void MainWindow::on_horizontalSlider_2_sliderMoved(int position)
{
    ui->label_5->setText(QString::number(position));
    int i,j;
    Mat img = Img;
    //cvtColor(img, img, COLOR_BGR2RGB);
    //QImage image = QImage((const unsigned char*)(image.data),image.cols,image.rows,QImage::Format_RGB888);
    QImage img2(img.cols*position, img.rows*position, QImage::Format_RGB888);

    for(i=0 ; i<img.rows*position ; i++)
    {
        for(j=0 ; j<img.cols*position ; j++)
        {

            int k = floor(i/position);
            int l = floor(j/position);
            int r = img.at<Vec3b>(k,l)[2];
            int g = img.at<Vec3b>(k,l)[1];
            int b = img.at<Vec3b>(k,l)[0];
            img2.setPixel(j,i,qRgb(r,g,b));

        }
    }
    ui->label->setPixmap(QPixmap::fromImage(img2));
//    ui->label->setPixmap(
//        QPixmap::fromImage(
//            QImage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888)
//            )
//        );
}


void MainWindow::on_horizontalSlider_3_sliderMoved(int position)
{
    ui->label_6->setText(QString::number(position));
    int i,j;
    Mat img = Img;
    //cvtColor(img, img, COLOR_BGR2RGB);
    //QImage image = QImage((const unsigned char*)(image.data),image.cols,image.rows,QImage::Format_RGB888);
    int q = floor(img.rows/position);
    int r = floor(img.cols/position);
    QImage img2(r, q, QImage::Format_RGB888);

    for(i=0 ; i<q ; i++)
    {
        for(j=0 ; j<r ; j++)
        {

            int k = i*position;
            int l = j*position;
            int r = img.at<Vec3b>(k,l)[2];
            int g = img.at<Vec3b>(k,l)[1];
            int b = img.at<Vec3b>(k,l)[0];
            img2.setPixel(j,i,qRgb(r,g,b));

        }
    }
    ui->label->setPixmap(QPixmap::fromImage(img2));
}


void MainWindow::on_horizontalSlider_4_sliderMoved(int position)
{
    ui->label_10->setText(QString::number(position));
    int i,j;
    Mat img = Img;
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);
    for( i=0; i<256; i++ ) histogram[i] = 0;
    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {

            int m = img.at<Vec3b>(i,j)[0]*0.114 + img.at<Vec3b>(i,j)[1]*0.587 + img.at<Vec3b>(i,j)[2]*0.299;
            m = floor(m*position/256)*(256/position);
            img2.setPixel(j,i,qRgb(m,m,m));
            histogram[ m ]++;

        }
    }
    ui->label_2->setPixmap(QPixmap::fromImage(img2));

    QBarSet *set0 = new QBarSet("Histogram");

    for(i=0;i<256;i++)
        *set0 << histogram[i];

    QBarSeries *series = new QBarSeries();
    series->append(set0);

    QChart *chart = new QChart();
    chart->addSeries(series);
    chart->setTitle("Histogram");
    chart->setAnimationOptions(QChart::SeriesAnimations);
    chart->legend()->setVisible(false);
    chart->legend()->setAlignment(Qt::AlignBottom);

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    while(!ui->horizontalLayout->isEmpty())
    {
        // Clear the horizontal layout content if there is any
        ui->horizontalLayout->removeItem(ui->horizontalLayout->itemAt(0));
    }
    ui->horizontalLayout->addWidget(chartView);
}

float alpha0 = 1;
int beta0 = 0;
void MainWindow::on_horizontalSlider_5_sliderMoved(int position)
{
    ui->label_12->setText(QString::number(position));
    int i,j;
    alpha0 = 1 + 0.1 * position;
    Mat img = Img;
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);
    //for( i=0; i<256; i++ ) histogram[i] = 0;
    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {
            // Set the pixel value of the QImage
            //QImage img3 = img;
            //QColor c = img.pixel(i,j);
            int r = floor(img.at<Vec3b>(i,j)[2]*alpha0+beta0);
            int g = floor(img.at<Vec3b>(i,j)[1]*alpha0+beta0);
            int b = floor(img.at<Vec3b>(i,j)[0]*alpha0+beta0);
            r = qBound(0,r,255);
            g = qBound(0,g,255);
            b = qBound(0,b,255);
            img2.setPixel(j,i,qRgb(r,g,b));
            //histogram[ m ]++;
        }
    }
    ui->label->setPixmap(QPixmap::fromImage(img2));
}


void MainWindow::on_horizontalSlider_6_sliderMoved(int position)
{
    ui->label_13->setText(QString::number(position));
    int i,j;
    beta0 = position;
    Mat img = Img;
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);
    //for( i=0; i<256; i++ ) histogram[i] = 0;
    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {
            // Set the pixel value of the QImage
            //QImage img3 = img;
            //QColor c = img.pixel(i,j);
            int r = floor(img.at<Vec3b>(i,j)[2]*alpha0+beta0);
            int g = floor(img.at<Vec3b>(i,j)[1]*alpha0+beta0);
            int b = floor(img.at<Vec3b>(i,j)[0]*alpha0+beta0);
            r = qBound(0,r,255);
            g = qBound(0,g,255);
            b = qBound(0,b,255);
            img2.setPixel(j,i,qRgb(r,g,b));
            //histogram[ m ]++;
        }
    }
    ui->label->setPixmap(QPixmap::fromImage(img2));
}

int rhistogram[256];
int ghistogram[256];
int bhistogram[256];
int rcdf[256];
int gcdf[256];
int bcdf[256];

void MainWindow::on_pushButton_4_clicked()
{
    int i,j;
    Mat img = Img;
    QImage img2(img.cols, img.rows, QImage::Format_RGB888);

    for( i=0; i<256; i++ ) rhistogram[i] = 0,ghistogram[i] = 0,bhistogram[i] = 0,rcdf[i] = 0,gcdf[i] = 0,bcdf[i] = 0;

    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {
            // Set the pixel value of the QImage
            //QImage img3 = img;
            //QColor c = img.pixel(i,j);
            //int r = (img.at<Vec3b>(i,j)[0] + img.at<Vec3b>(i,j)[1] + img.at<Vec3b>(i,j)[2])/3;
            //img2.setPixel(j,i,qRgb(m,m,m));
            rhistogram[ img.at<Vec3b>(i,j)[2] ]++;
            ghistogram[ img.at<Vec3b>(i,j)[1] ]++;
            bhistogram[ img.at<Vec3b>(i,j)[0] ]++;
        }
    }
    rcdf[0] = rhistogram[0];
    gcdf[0] = ghistogram[0];
    bcdf[0] = bhistogram[0];
    for (i = 1; i < 256; i ++)
    {
        rcdf[i] = rcdf[i - 1] + rhistogram[i];
        gcdf[i] = gcdf[i - 1] + ghistogram[i];
        bcdf[i] = bcdf[i - 1] + bhistogram[i];
    }

    int max = img.rows*img.cols;

    for(i=0 ; i<img.rows ; i++)
    {
        for(j=0 ; j<img.cols ; j++)
        {

            float x = rcdf[img.at<Vec3b>(i,j)[2]];
            float y = gcdf[img.at<Vec3b>(i,j)[1]];
            float z = bcdf[img.at<Vec3b>(i,j)[0]];
            float rf = (x - rcdf[0]) / (max-rcdf[0]);
            float gf = (y - gcdf[0]) / (max-gcdf[0]);
            float bf = (z - bcdf[0]) / (max-bcdf[0]);
            int r = floor(rf*255);
            int g = floor(gf*255);
            int b = floor(bf*255);
            img2.setPixel(j,i,qRgb(r,g,b));
            //histogram[ m ]++;



        }
    }
    ui->label->setPixmap(QPixmap::fromImage(img2));

}

