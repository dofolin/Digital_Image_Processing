#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "opencv2/opencv.hpp"
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
QString Img1;
void MainWindow::on_actionopen_file1_triggered()
{
    char a;
    int i,j;

    // Open file dialog to get the file name
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.64)"));
    Img1 = fileName;
    // Read the .64 text file and convert the characters into image array
    if(fileName != NULL)
    {
        ifstream imagefile;
        imagefile.open(fileName.toStdString());

        if (imagefile.is_open())
        {
            for(i=0; i<64; i++)
            {
                for(j=0; j<64; j++)
                {
                    // Get the character and convert it to integer ranging from 0 to 255
                    image[i][j] = imagefile.get();
                    // cout <<image[i][j]; // Use this line to check if the input is correct

                    if( (image[i][j] >= '0') && (image[i][j] <= '9') )
                    {
                        image[i][j] = (image[i][j]-'0');
                    }
                    else
                    {
                        image[i][j] = ((image[i][j]-'A') + 10);
                    }
                }
                a = imagefile.get();  // Discard the end of line character
                // cout << a;  // Use this line to check if the input is correct
            }
            imagefile.close();
        }

        // Set up QImage for displaying it in the QLabel label

        QImage img(64, 64, QImage::Format_RGB32);
        for(i=0;i<64;i++)
        {
            for(j=0;j<64;j++)
            {
                // Set the pixel value of the QImage
                img.setPixel(j,i,qRgb(image[i][j]*8,image[i][j]*8,image[i][j]*8));
            }
        }

        // Display QImage on the label
        ui->label->setPixmap(QPixmap::fromImage(img));
        // Resize the QImage to fit the label display
        QImage imgResize = img.scaled(ui->label->width(),ui->label->height(),Qt::KeepAspectRatio);
        ui->label->setPixmap(QPixmap::fromImage(imgResize));


        // Display QImage on the label
        ui->label_3->setPixmap(QPixmap::fromImage(img));
        // Resize the QImage to fit the label display
        //QImage imgResize = img.scaled(ui->label_3->width(),ui->label_3->height(),Qt::KeepAspectRatio);
        ui->label_3->setPixmap(QPixmap::fromImage(imgResize));


        // Calculate the histogram of the 64x64 image;
        for( i=0; i<32; i++ ) histogram[i] = 0;	/* Initialize the array */
        for(i=0; i<64; i++)
        {
            for(j=0; j<64; j++)
            {
                histogram[ image[i][j] ]++;
            }
        }


        // Use QChart to display the image histogram

        QBarSet *set0 = new QBarSet("Histogram");

        for(i=0;i<32;i++)
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

}

QString Img2;
void MainWindow::on_actionopen_file2_triggered()
{
    char a;
    int i,j;

    // Open file dialog to get the file name
    QString fileName = QFileDialog::getOpenFileName(this,
                                                    tr("Open Image"), ".",
                                                    tr("Image Files (*.64)"));
    //Mat img =cv::imread(fileName.toStdString());
    Img2 = fileName;
    // Read the .64 text file and convert the characters into image array
    if(fileName != NULL)
    {
        ifstream imagefile;
        imagefile.open(fileName.toStdString());

        if (imagefile.is_open())
        {
            for(i=0; i<64; i++)
            {
                for(j=0; j<64; j++)
                {
                    // Get the character and convert it to integer ranging from 0 to 255
                    image[i][j] = imagefile.get();
                    // cout <<image[i][j]; // Use this line to check if the input is correct

                    if( (image[i][j] >= '0') && (image[i][j] <= '9') )
                    {
                        image[i][j] = (image[i][j]-'0');
                    }
                    else
                    {
                        image[i][j] = ((image[i][j]-'A') + 10);
                    }
                }
                a = imagefile.get();  // Discard the end of line character
                // cout << a;  // Use this line to check if the input is correct
            }
            imagefile.close();
        }

        // Set up QImage for displaying it in the QLabel label

        QImage img(64, 64, QImage::Format_RGB32);
        for(i=0;i<64;i++)
        {
            for(j=0;j<64;j++)
            {
                // Set the pixel value of the QImage
                img.setPixel(j,i,qRgb(image[i][j]*8,image[i][j]*8,image[i][j]*8));
            }
        }

        // Display QImage on the label
        ui->label_2->setPixmap(QPixmap::fromImage(img));
        // Resize the QImage to fit the label display
        QImage imgResize = img.scaled(ui->label_2->width(),ui->label_2->height(),Qt::KeepAspectRatio);
        ui->label_2->setPixmap(QPixmap::fromImage(imgResize));


        // Calculate the histogram of the 64x64 image;
        for( i=0; i<32; i++ ) histogram[i] = 0;	/* Initialize the array */
        for(i=0; i<64; i++)
        {
            for(j=0; j<64; j++)
            {
                histogram[ image[i][j] ]++;
            }
        }


        // Use QChart to display the image histogram

        QBarSet *set0 = new QBarSet("Histogram");

        for(i=0;i<32;i++)
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

        while(!ui->horizontalLayout_2->isEmpty())
        {
            // Clear the horizontal layout content if there is any
            ui->horizontalLayout_2->removeItem(ui->horizontalLayout_2->itemAt(0));
        }
        ui->horizontalLayout_2->addWidget(chartView);

    }

}


void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    ui->label_4->setText(QString::number(position));

    char a;
    int i,j;
    QString fileName = Img1;
    // Read the .64 text file and convert the characters into image array
    if(fileName != NULL)
    {
        ifstream imagefile;
        imagefile.open(fileName.toStdString());

        if (imagefile.is_open())
        {
            for(i=0; i<64; i++)
            {
                for(j=0; j<64; j++)
                {
                    // Get the character and convert it to integer ranging from 0 to 255
                    image[i][j] = imagefile.get();
                    // cout <<image[i][j]; // Use this line to check if the input is correct

                    if( (image[i][j] >= '0') && (image[i][j] <= '9') )
                    {
                        image[i][j] = (image[i][j]-'0');
                    }
                    else
                    {
                        image[i][j] = ((image[i][j]-'A') + 10);
                    }
                }
                a = imagefile.get();  // Discard the end of line character
                // cout << a;  // Use this line to check if the input is correct
            }
            imagefile.close();
        }

//     Set up QImage for displaying it in the QLabel label

        QImage img(64, 64, QImage::Format_RGB32);
        for(i=0;i<64;i++)
        {
            for(j=0;j<64;j++)
            {
                // Set the pixel value of the QImage
                int x = (image[i][j] + position)*8;
                x = qBound(0,x,255);
                img.setPixel(j,i,qRgb(x,x,x));
            }
        }

        // Display QImage on the label
        ui->label_3->setPixmap(QPixmap::fromImage(img));
        // Resize the QImage to fit the label display
        QImage imgResize = img.scaled(ui->label_3->width(),ui->label_3->height(),Qt::KeepAspectRatio);
        ui->label_3->setPixmap(QPixmap::fromImage(imgResize));


        // Calculate the histogram of the 64x64 image;
        for( i=0; i<32; i++ ) histogram[i] = 0;	/* Initialize the array */
        for(i=0; i<64; i++)
        {
            for(j=0; j<64; j++)
            {
                int x = image[i][j] + position;
                x = qBound(0,x,31);
                histogram[ x ]++;
            }
        }


        // Use QChart to display the image histogram

        QBarSet *set0 = new QBarSet("Histogram");

        for(i=0;i<32;i++)
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

        while(!ui->horizontalLayout_3->isEmpty())
        {
            // Clear the horizontal layout content if there is any
            ui->horizontalLayout_3->removeItem(ui->horizontalLayout_3->itemAt(0));
        }
        ui->horizontalLayout_3->addWidget(chartView);
    }

}


void MainWindow::on_horizontalSlider_2_sliderMoved(int position)
{
    ui->label_5->setText(QString::number(position));

    char a;
    int i,j;
    QString fileName = Img1;
    // Read the .64 text file and convert the characters into image array
    if(fileName != NULL)
    {
        ifstream imagefile;
        imagefile.open(fileName.toStdString());

        if (imagefile.is_open())
        {
            for(i=0; i<64; i++)
            {
                for(j=0; j<64; j++)
                {
                    // Get the character and convert it to integer ranging from 0 to 255
                    image[i][j] = imagefile.get();
                    // cout <<image[i][j]; // Use this line to check if the input is correct

                    if( (image[i][j] >= '0') && (image[i][j] <= '9') )
                    {
                        image[i][j] = (image[i][j]-'0');
                    }
                    else
                    {
                        image[i][j] = ((image[i][j]-'A') + 10);
                    }
                }
                a = imagefile.get();  // Discard the end of line character
                // cout << a;  // Use this line to check if the input is correct
            }
            imagefile.close();
        }

        //     Set up QImage for displaying it in the QLabel label

        QImage img(64, 64, QImage::Format_RGB32);
        for(i=0;i<64;i++)
        {
            for(j=0;j<64;j++)
            {
                // Set the pixel value of the QImage
                int x = (image[i][j] * position * 8);
                x = qBound(0,x,255);
                img.setPixel(j,i,qRgb(x,x,x));
            }
        }

        // Display QImage on the label
        ui->label_3->setPixmap(QPixmap::fromImage(img));
        // Resize the QImage to fit the label display
        QImage imgResize = img.scaled(ui->label_3->width(),ui->label_3->height(),Qt::KeepAspectRatio);
        ui->label_3->setPixmap(QPixmap::fromImage(imgResize));


        // Calculate the histogram of the 64x64 image;
        for( i=0; i<32; i++ ) histogram[i] = 0;	/* Initialize the array */
        for(i=0; i<64; i++)
        {
            for(j=0; j<64; j++)
            {
                int x = (image[i][j] * position);
                x = qBound(0,x,31);
                histogram[ x ]++;
            }
        }


        // Use QChart to display the image histogram

        QBarSet *set0 = new QBarSet("Histogram");

        for(i=0;i<32;i++)
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

        while(!ui->horizontalLayout_3->isEmpty())
        {
            // Clear the horizontal layout content if there is any
            ui->horizontalLayout_3->removeItem(ui->horizontalLayout_3->itemAt(0));
        }
        ui->horizontalLayout_3->addWidget(chartView);
    }

}
unsigned char image2[64][64];

void MainWindow::on_pushButton_clicked()
{
    int i,j;
    char a;

    QString fileName = Img1;
    QString fileName2 = Img2;
    if(fileName != NULL && fileName2 != NULL)
    {
        ifstream imagefile;
        imagefile.open(fileName.toStdString());

        if (imagefile.is_open())
        {
            for(i=0; i<64; i++)
            {
                for(j=0; j<64; j++)
                {
                    // Get the character and convert it to integer ranging from 0 to 255
                    image[i][j] = imagefile.get();
                    // cout <<image[i][j]; // Use this line to check if the input is correct

                    if( (image[i][j] >= '0') && (image[i][j] <= '9') )
                    {
                        image[i][j] = (image[i][j]-'0');
                    }
                    else
                    {
                        image[i][j] = ((image[i][j]-'A') + 10);
                    }
                }
                a = imagefile.get();  // Discard the end of line character
                // cout << a;  // Use this line to check if the input is correct
            }
            imagefile.close();
        }
        ifstream imagefile2;
        imagefile2.open(fileName2.toStdString());

        if (imagefile2.is_open())
        {
            for(i=0; i<64; i++)
            {
                for(j=0; j<64; j++)
                {
                    // Get the character and convert it to integer ranging from 0 to 255
                    image2[i][j] = imagefile2.get();
                    // cout <<image[i][j]; // Use this line to check if the input is correct

                    if( (image2[i][j] >= '0') && (image2[i][j] <= '9') )
                    {
                        image2[i][j] = (image2[i][j]-'0');
                        image[i][j] = (image[i][j] + image2[i][j])/2;
                    }
                    else
                    {
                        image2[i][j] = ((image2[i][j]-'A') + 10);
                        image[i][j] = (image[i][j] + image2[i][j])/2;
                    }
                }
                a = imagefile2.get();  // Discard the end of line character
                // cout << a;  // Use this line to check if the input is correct
            }
            imagefile2.close();
        }

        // Set up QImage for displaying it in the QLabel label

        QImage img(64, 64, QImage::Format_RGB32);
        for(i=0;i<64;i++)
        {
            for(j=0;j<64;j++)
            {
                // Set the pixel value of the QImage
                img.setPixel(j,i,qRgb(image[i][j]*8,image[i][j]*8,image[i][j]*8));
            }
        }

        // Display QImage on the label
        ui->label_3->setPixmap(QPixmap::fromImage(img));
        // Resize the QImage to fit the label display
        QImage imgResize = img.scaled(ui->label_3->width(),ui->label_3->height(),Qt::KeepAspectRatio);
        ui->label_3->setPixmap(QPixmap::fromImage(imgResize));

        // Calculate the histogram of the 64x64 image;
        for( i=0; i<32; i++ ) histogram[i] = 0;	/* Initialize the array */
        for(i=0; i<64; i++)
        {
            for(j=0; j<64; j++)
            {
                histogram[ image[i][j] ]++;
            }
        }


        // Use QChart to display the image histogram

        QBarSet *set0 = new QBarSet("Histogram");

        for(i=0;i<32;i++)
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

        while(!ui->horizontalLayout_3->isEmpty())
        {
            // Clear the horizontal layout content if there is any
            ui->horizontalLayout_3->removeItem(ui->horizontalLayout_3->itemAt(0));
        }
        ui->horizontalLayout_3->addWidget(chartView);

    }
}

void MainWindow::on_pushButton_2_clicked()
{
    char a;
    int i,j;
    QString fileName = Img1;
    QString fileName2= Img1;
    // Open file dialog to get the file name
//    QString fileName = QFileDialog::getOpenFileName(this,
//                                                    tr("Open Image"), ".",
//                                                    tr("Image Files (*.64)"));
//    Img1 = fileName;
    // Read the .64 text file and convert the characters into image array
    if(fileName != NULL && fileName2 != NULL)
    {
        ifstream imagefile;
        imagefile.open(fileName.toStdString());

        if (imagefile.is_open())
        {
            for(i=0; i<64; i++)
            {
                for(j=0; j<64; j++)
                {
                    // Get the character and convert it to integer ranging from 0 to 255
                    image[i][j] = imagefile.get();
                    // cout <<image[i][j]; // Use this line to check if the input is correct

                    if( (image[i][j] >= '0') && (image[i][j] <= '9') )
                    {
                        image[i][j] = (image[i][j]-'0');
                    }
                    else
                    {
                        image[i][j] = ((image[i][j]-'A') + 10);
                    }
                }
                a = imagefile.get();  // Discard the end of line character
                // cout << a;  // Use this line to check if the input is correct
            }
            imagefile.close();
        }

        ifstream imagefile2;
        imagefile2.open(fileName2.toStdString());

        if (imagefile2.is_open())
        {
            for(i=0; i<64; i++)
            {
                for(j=0; j<64; j++)
                {
                    // Get the character and convert it to integer ranging from 0 to 255
                    image2[i][j] = imagefile2.get();
                    // cout <<image[i][j]; // Use this line to check if the input is correct

                    if( (image2[i][j] >= '0') && (image2[i][j] <= '9') )
                    {
                        image2[i][j] = (image2[i][j]-'0');
                        //image2[i][j] = image[i][j] - image[i-1][j];
                    }
                    else
                    {
                        image2[i][j] = ((image2[i][j]-'A') + 10);
                        //image2[i][j] = image[i][j] - image[i-1][j];
                    }

                }
                a = imagefile2.get();  // Discard the end of line character
                // cout << a;  // Use this line to check if the input is correct
            }
            imagefile2.close();
        }

        // Set up QImage for displaying it in the QLabel label

        QImage img(64, 64, QImage::Format_RGB32);
        for(i=0;i<64;i++)
        {
            for(j=0;j<64;j++)
            {
                // Set the pixel value of the QImage
                int x = (image2[i][j]-image[i][j-1])*8;
                x = qBound(0,x,255);
                img.setPixel(j,i,qRgb(x,x,x));
            }
        }

        // Display QImage on the label
        ui->label_3->setPixmap(QPixmap::fromImage(img));
        // Resize the QImage to fit the label display
        QImage imgResize = img.scaled(ui->label_3->width(),ui->label_3->height(),Qt::KeepAspectRatio);
        ui->label_3->setPixmap(QPixmap::fromImage(imgResize));



        // Calculate the histogram of the 64x64 image;
        for( i=0; i<32; i++ ) histogram[i] = 0;	/* Initialize the array */
        for(i=0; i<64; i++)
        {
            for(j=0; j<64; j++)
            {
                int x = (image2[i][j]-image[i][j-1]);
                x = qBound(0,x,31);
                histogram[ x ]++;
            }
        }


        // Use QChart to display the image histogram

        QBarSet *set0 = new QBarSet("Histogram");

        for(i=0;i<32;i++)
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

        while(!ui->horizontalLayout_3->isEmpty())
        {
            // Clear the horizontal layout content if there is any
            ui->horizontalLayout_3->removeItem(ui->horizontalLayout_3->itemAt(0));
        }
        ui->horizontalLayout_3->addWidget(chartView);



    }
}

