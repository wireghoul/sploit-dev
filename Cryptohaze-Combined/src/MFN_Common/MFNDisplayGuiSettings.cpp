#include "MFN_Common/MFNDisplayGuiSettings.h"

#include "ui_MFNDisplayGuiSettings.h"

#include <QFileDialog>

// For OpenCL device count
#include "OpenCL_Common/GRTOpenCL.h"

MFNDisplayGuiSettings::MFNDisplayGuiSettings(QWidget *parent) :
    QDialog(parent)
{
    ui.setupUi(this);

    setWindowFlags(Qt::Window | Qt::WindowTitleHint | Qt::CustomizeWindowHint | Qt::WindowMinimizeButtonHint | Qt::WindowCloseButtonHint);

    //setFixedSize(569, 436);

    ui.comboBoxHashType->addItem("MD5");
    ui.comboBoxHashType->addItem("NTLM");

    signalMapper = new QSignalMapper(this);
    connectPushButtonToLineEdit(ui.pushButtonCharsetFile, ui.lineEditCharsetFile);
    connectPushButtonToLineEdit(ui.pushButtonHashFile, ui.lineEditHashFile);
    connectPushButtonToLineEdit(ui.pushButtonNotFoundFile, ui.lineEditNotFoundFile);
    connectPushButtonToLineEdit(ui.pushButtonOutputFile, ui.lineEditOutputFile);
    connectPushButtonToLineEdit(ui.pushButtonRestoreFile, ui.lineEditRestoreFile);
    connect(signalMapper, SIGNAL(mapped(QWidget*)), this, SLOT(pushButton_clicked(QWidget *)));
}

MFNDisplayGuiSettings::~MFNDisplayGuiSettings()
{
    delete signalMapper;
}

void MFNDisplayGuiSettings::connectPushButtonToLineEdit(QPushButton *button, QLineEdit *edit)
{
    connect(button, SIGNAL(clicked()), signalMapper, SLOT(map()));
    signalMapper->setMapping(button, edit);
}

void MFNDisplayGuiSettings::pushButton_clicked(QWidget *edit)
{
    QString fileName = QFileDialog::getOpenFileName(this, "Select File");

    if(!fileName.isNull())
    {
        ((QLineEdit *)edit)->setText(fileName);
    }
}

std::vector<MFNDeviceInformation> MFNDisplayGuiSettings::GetDevicesToUse()
{
    std::vector<MFNDeviceInformation> devices;

    MFNDeviceInformation DeviceInfo;
    uint32_t OpenCL_Platform = 0;
    uint32_t OpenCL_Thread_Count = 0;
    uint32_t CUDA_Thread_Count = 0;
    int numberCudaDevices = 0;
    int device;
        
    // Handle CUDA threads
    if (ui.checkBoxUseCUDA->isChecked()) {
        memset(&DeviceInfo, 0, sizeof(MFNDeviceInformation));
        DeviceInfo.IsCUDADevice = 1;
        cudaGetDeviceCount(&numberCudaDevices);
        /*
        if (this->Debug || this->DevDebug) {
                printf("Got %d CUDA devices!\n", numberCudaDevices);
        }
        */

        // Just use them all.
        for (device = 0; device < numberCudaDevices; device++) {
            DeviceInfo.GPUDeviceId = device;
            devices.push_back(DeviceInfo);
            CUDA_Thread_Count++;
        }
    }
        
    // OpenCL Devices
    if (ui.checkBoxUseOpenCL->isChecked()) {
        memset(&DeviceInfo, 0, sizeof(MFNDeviceInformation));
        DeviceInfo.IsOpenCLDevice = 1;
        DeviceInfo.OpenCLPlatformId = OpenCL_Platform;
            
        // Enumerate the OpenCL devices & add them all.
        CryptohazeOpenCL OpenCL;
        int numberOpenCLGPUs;

        OpenCL.getNumberOfPlatforms();
        OpenCL.selectPlatformById(OpenCL_Platform);
        OpenCL.setDeviceType(DEVICE_ALL);
        numberOpenCLGPUs = OpenCL.getNumberOfDevices();
        /*
        if (this->Debug || this->DevDebug) {
            printf("Got %d OpenCL GPUs\n", numberOpenCLGPUs);
        }
        */
        for (device = 0; device < numberOpenCLGPUs; device++) {
            DeviceInfo.GPUDeviceId = device;
            devices.push_back(DeviceInfo);
            OpenCL_Thread_Count++;
        }
    }

        
    if (ui.checkBoxUseCPU->isChecked()) {
        memset(&DeviceInfo, 0, sizeof(MFNDeviceInformation));
        /*
        if (this->Debug || this->DevDebug) {
            printf("Got %d CPU cores\n", (int)boost::thread::hardware_concurrency());
        }
        */

        DeviceInfo.IsCPUDevice = 1;
        // If there are cores left, add them.
        if (((int)boost::thread::hardware_concurrency() 
                - (int)OpenCL_Thread_Count - (int)CUDA_Thread_Count) > 0) {
            DeviceInfo.DeviceThreads = (int)boost::thread::hardware_concurrency() 
                    - (int)OpenCL_Thread_Count - (int)CUDA_Thread_Count;
            devices.push_back(DeviceInfo);
        }
    }

    return devices;
}

