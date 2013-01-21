#include "MFN_Common/MFNDisplayGui.h"
#include "MFN_Common/MFNDefines.h"
#include "MFN_Common/MFNMultiforcerClassFactory.h"

#include "MFN_Common/MFNRun.h"

#include <QtGui>
#include <boost/thread.hpp>

extern struct global_commands global_interface;
extern MFNClassFactory MultiforcerGlobalClassFactory;

MFNDisplayGui::MFNDisplayGui(QWidget *parent)
    : QMainWindow(parent), settings(parent)
{
    ui.setupUi(this);

    shownMinimizedBalloon = false;
    connect(ui.actionSettings, SIGNAL(triggered()), this, SLOT(showSettingsClicked()));

    createControlButtonbox();
    createActions();
    createTray();

    connect(trayIcon, SIGNAL(activated(QSystemTrayIcon::ActivationReason)),
             this, SLOT(trayActivated(QSystemTrayIcon::ActivationReason)));

    connect(ui.actionSettingsExit, SIGNAL(triggered()), this, SLOT(exitApp()));

    labelNetClients = new QLabel("N/A");
    labelWorkUnits = new QLabel("N/A");

    isRunning = false;

    this->setSystemMode(SYSTEM_MODE_STANDALONE, "Standalone");
    numNetClients = 0;

    MultiforcerGlobalClassFactory.setDisplayClass(this);
    MultiforcerGlobalClassFactory.setDisplayClassType(MFN_DISPLAY_CLASS_GUI);
    MultiforcerGlobalClassFactory.setCommandlinedataClass(&settings);
    

    lastTimeUpdated = 0;

    trayIcon->show();
}


void MFNDisplayGui::Refresh()  
{
    // Check to see if it's been long enough that we can update.
    if ((this->displayTimer.getElapsedTime() - this->lastTimeUpdated) < 0.5) {
        return;
    }

    this->statusUpdateMutexBoost.lock();
    this->displayUpdateMutex.lock();

    CHHashFileV *hashFileClass = MultiforcerGlobalClassFactory.getHashfileClass();;
    MFNWorkunitBase *workunitClass = MultiforcerGlobalClassFactory.getWorkunitClass();


    uint32_t tmpWUsCompleted = workunitClass->GetNumberOfCompletedWorkunits();
    if(tmpWUsCompleted != workunitsCompleted)
    {
        workunitsCompleted = tmpWUsCompleted;
    }

    uint32_t tmpWUsTotal = workunitClass->GetNumberOfWorkunits();
    if(workunitsTotal != tmpWUsTotal)
    {
        workunitsTotal = tmpWUsTotal;
    }

    uint64_t tmpCrackedHashes =  hashFileClass->GetCrackedHashCount();
    if(crackedHashes != tmpCrackedHashes)
    {
        crackedHashes = tmpCrackedHashes;
        ui.labelCrackedHashes->setText(QString::number(crackedHashes));
    }

    uint64_t tmpTotalHashes = hashFileClass->GetTotalHashCount();
    if(totalHashes != tmpTotalHashes)
    {
        totalHashes = tmpTotalHashes;
        ui.labelTotalHashes->setText(QString::number(totalHashes));
    }

    uint64_t crackingTimeSeconds = (uint64_t) displayTimer.getElapsedTime();

    int seconds = crackingTimeSeconds % 60;
    int minutes = (crackingTimeSeconds / 60) % 60;
    int hours = crackingTimeSeconds / 3600;
    QString time;
    time.sprintf("%02d:%02d:%02d", hours, minutes, seconds);
    ui.labelTotalTime->setText(time);

    if(systemMode != SYSTEM_MODE_CLIENT)
    {
        QString wulabel;
        wulabel = QString::number(workunitsCompleted);
        wulabel += "/";
        wulabel += QString::number(workunitsTotal);
        wulabel += " (";
        wulabel += QString::number((float)workunitsCompleted/(float)workunitsTotal, 'g', 2);
        wulabel += "%)";

        labelWorkUnits->setText(wulabel);
    }

    float totalSpeed = 0;
    QFormLayout *deviceStatusLayout = ((QFormLayout *) ui.groupBoxDevices->layout());
    QString label;
    QString value;
    int i;
    for(i = 0; i < threadType.size(); ++i)
    {
        QString status;
        if(threadType[i] == GPU_THREAD)
        {
            label = QString::number(i) + " GPU:";
        }
        else if(threadType[i] == CPU_THREAD)
        {
            label = QString::number(i) + " CPU:";
        }
        else if(threadType[i] == NETWORK_HOST)
        {
            label = QString::number(i) + " NET:";
        }
        value = QString(this->getConvertedRateString(this->threadRate[i]).c_str()) + "/s";

        ((QLabel *) deviceStatusLayout->itemAt(i + 1, 
            QFormLayout::LabelRole)->widget())->setText(
            label
            );

        ((QLabel *) deviceStatusLayout->itemAt(i + 1, 
            QFormLayout::FieldRole)->widget())->setText(
            value
            );

        totalSpeed += threadRate[i];
    }

    ((QLabel *) deviceStatusLayout->itemAt(i + 1, 
        QFormLayout::LabelRole)->widget())->setText(
        "TOTAL:"
        );
    ((QLabel *) deviceStatusLayout->itemAt(i + 1, 
        QFormLayout::FieldRole)->widget())->setText(
        QString::fromStdString(getConvertedRateString(totalSpeed)) + "/s"
        );

    lastTimeUpdated = displayTimer.getElapsedTime();

    this->statusUpdateMutexBoost.unlock();
    this->displayUpdateMutex.unlock();
}
    
/**
    * Sets the currently-active hash name.
    * @param newHashName String of the hash name.
    */
void MFNDisplayGui::setHashName(std::string newHashName) 
{ 
    this->statusUpdateMutexBoost.lock();
    ui.labelHashType->setText(QString::fromStdString(newHashName));
    this->statusUpdateMutexBoost.unlock();
}

/**
    * Sets the current password length being cracked.
    * @param newPasswordLength New length.
    */
void MFNDisplayGui::setPasswordLen(uint16_t newPasswordLength) 
{ 
    this->statusUpdateMutexBoost.lock();
    ui.labelCurrentPWLen->setText(QString::number(newPasswordLength));
    this->statusUpdateMutexBoost.unlock();
}
      
/**
    * Adds a new cracked password.  This is a vector, as that is how the
    * passwords are handled internally.
    * @param newFoundPassword A vector containing the password string.
    */
void MFNDisplayGui::addCrackedPassword(std::vector<uint8_t> newFoundPassword) 
{ 
    char buffer[256];
    memset(buffer, 0, sizeof(buffer));
    memcpy(buffer, &newFoundPassword[0], newFoundPassword.size());

    std::string crackedPassword(buffer);

    this->statusUpdateMutexBoost.lock();
    ui.textEditCrackedHashes->append(QString::fromStdString(crackedPassword));
    this->statusUpdateMutexBoost.unlock();
}

/**
    * Adds a new status line to the system.
    * @param newStatusLine std::string or char* status line.
    */
void MFNDisplayGui::addStatusLine(std::string newStatusLine) 
{
    this->statusUpdateMutexBoost.lock();
    ui.textEditStatus->append(QString::fromStdString(newStatusLine));
    this->statusUpdateMutexBoost.unlock();
}

void MFNDisplayGui::addStatusLine(char * newStatusLine) 
{
    addStatusLine(std::string(newStatusLine));
}

// Sets the system mode: Standalone, network server, network client.
void MFNDisplayGui::setSystemMode(int systemMode, std::string modeString) 
{
    this->statusUpdateMutexBoost.lock();
    this->systemMode = systemMode;
    ui.labelMode->setText(QString::fromStdString(modeString));

    /* TODO - check to see if we've already added these to devices... */
    if(this->systemMode == SYSTEM_MODE_SERVER)
    {
        ((QFormLayout *) ui.groupBoxDevices->layout())->addRow(
            new QLabel("Net clients:"),
            labelNetClients
            );
    }

    if(this->systemMode != SYSTEM_MODE_CLIENT)
    {
        ((QFormLayout *) ui.groupBoxDevices->layout())->addRow(
            new QLabel("WUs:"),
            labelWorkUnits
            );
    }
    this->statusUpdateMutexBoost.unlock();
}

// Add or subtract from the number of connected clients
void MFNDisplayGui::alterNetworkClientCount(int networkClientCount) 
{
    this->statusUpdateMutexBoost.lock();
    this->numNetClients += networkClientCount;
    labelNetClients->setText(QString::number(this->numNetClients));
    this->statusUpdateMutexBoost.unlock();
}


void MFNDisplayGui::createControlButtonbox()
{
    pushButtonStart = new QPushButton("Start");
    connect(pushButtonStart, SIGNAL(clicked()), this, SLOT(startButtonClicked()));

    QDialogButtonBox *bbox = new QDialogButtonBox(Qt::Horizontal);
    bbox->addButton(pushButtonStart, QDialogButtonBox::ActionRole);

    ui.centralwidget->layout()->addWidget(bbox);

}

void MFNDisplayGui::createTray()
{
    trayIconMenu = new QMenu(this);
    trayIconMenu->addAction(showSettingsAction);
    trayIconMenu->addSeparator();
    trayIconMenu->addAction(restoreAction);
    trayIconMenu->addAction(minimizeAction);
    trayIconMenu->addSeparator();
    trayIconMenu->addAction(quitAction);

    trayIcon = new QSystemTrayIcon(this);
    trayIcon->setContextMenu(trayIconMenu);

    // TODO - replace with non-stock qt example icon
    trayIcon->setIcon(QIcon(":/MFNDisplayGuiTrayIcon.svg"));
    trayIcon->setToolTip("Cryptohaze Multiforcer");
}

void MFNDisplayGui::createActions()
{
    minimizeAction = new QAction("Mi&nimize", this);
    connect(minimizeAction, SIGNAL(triggered()), this, SLOT(hide()));

    restoreAction = new QAction("&Restore", this);
    connect(restoreAction, SIGNAL(triggered()), this, SLOT(showNormal()));

    showSettingsAction = new QAction("&Settings", this);
    connect(showSettingsAction, SIGNAL(triggered()), this, SLOT(showSettingsClicked()));

    quitAction = new QAction("&Quit", this);
    connect(quitAction, SIGNAL(triggered()), qApp, SLOT(quit()));
}

void MFNDisplayGui::showMinimizedBalloon()
{
    if(!shownMinimizedBalloon)
    {
        shownMinimizedBalloon = true;
        trayIcon->showMessage("Cryptohaze Multiforcer", 
            "Cryptohaze Multiforcer is still running, just minimized to system tray",
            QSystemTrayIcon::Information,
            1000
            );        
    }
}

void MFNDisplayGui::checkIdleTime()
{
    LASTINPUTINFO lastInputInfo;
    while(global_interface.exit == 0)
    {
        if(GetLastInputInfo(&lastInputInfo))
        {
            DWORD now = GetTickCount();

            if(global_interface.pause == 0 
               && ((now - lastInputInfo.dwTime) > 5 * 1000 * 60))
            {
                global_interface.pause = 1;
            }
            else if(global_interface.pause == 1)
            {
                global_interface.pause = 0;
                break;
            }
        }
        Sleep(1 * 1000);
    }
}

void MFNDisplayGui::showSettingsClicked()
{
    if(settings.exec() == QDialog::Accepted)
    {
        //ui.labelSettings1->setText(settings.GetHashFilename());
        //ui.labelSettings2->setText(settings.GetCharsetFilename());
    }
}

void MFNDisplayGui::startButtonClicked()
{
    this->statusUpdateMutexBoost.lock();
    if(!isRunning)
    {
        isRunning = true;
        pushButtonStart->setText("Pause");

        if(global_interface.pause == 1)
        {
            global_interface.pause = 0;
        }
        else
        {
            size_t numDevices = settings.GetDevicesToUse().size();
            // Need room for the total speed in the device list
            numDevices += 1;
            for(size_t i = 0; i < numDevices; ++i)
            {
                ((QFormLayout *) ui.groupBoxDevices->layout())->addRow(
                    new QLabel(QString::number(i)),
                    new QLabel("")
                    );
            }

            displayTimer.start();

            new boost::thread(MFNRun);
        }
    }
    else
    {

        isRunning = false;
        pushButtonStart->setText("Start");

        if(global_interface.pause == 0)
            global_interface.pause = 1;

        lastTimeUpdated = 0;
        displayTimer.stop();
    }
    this->statusUpdateMutexBoost.unlock();
}

void MFNDisplayGui::exitApp()
{
    global_interface.exit = 1;
    global_interface.user_exit = 1;

    this->trayIcon->hide();

    this->close();
}


void MFNDisplayGui::trayActivated(QSystemTrayIcon::ActivationReason reason)
{
    if(reason == QSystemTrayIcon::DoubleClick)
    {
        if(isHidden())
        {
            showNormal();
        }
        else
        {
            showMinimizedBalloon();
            hide();
        }
    }
}

void MFNDisplayGui::closeEvent(QCloseEvent *event)
{
    if (trayIcon->isVisible()) 
    {
        showMinimizedBalloon();        
        hide();
        event->ignore();
    }
}
