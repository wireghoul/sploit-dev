#ifndef __MFN_DISPLAY_GUI_SETTINGS_H
#define __MFN_DISPLAY_GUI_SETTINGS_H

#include "ui_MFNDisplayGuiSettings.h"

#include "MFN_Common/MFNCommandLineData.h"

#include <QDialog>
#include <QSignalMapper>
#include <QPushButton>
#include <QLineEdit>
#include <QString>

#include <stdint.h>
#include <string>
#include <vector>

class MFNDisplayGuiSettings : public QDialog, public MFNCommandLineData
{
    Q_OBJECT
    
public:
    MFNDisplayGuiSettings(QWidget *parent = 0);
    virtual ~MFNDisplayGuiSettings();

    QString GetHashFilename() { return ui.lineEditHashFile->text(); }
    QString GetCharsetFilename() { return ui.lineEditCharsetFile->text(); }



    uint32_t GetHashType() {
        return 0;
        //return this->HashType;
    }
    std::string GetHashTypeString()
    {
        return ui.comboBoxHashType->currentText().toStdString();
    }

    std::string GetHashListFileName() {
        return getString(ui.lineEditHashFile);
    }

    std::string GetCharsetFileName() {
        return getString(ui.lineEditCharsetFile);
    }
    char GetUseCharsetMulti() {
        return ui.checkBoxCharsetMulti->isChecked();
    }

    std::string GetOutputFileName() {
        return getString(ui.lineEditOutputFile);
    }

    std::string GetUnfoundOutputFileName() {
        return getString(ui.lineEditNotFoundFile);
    }

    std::string GetRestoreFileName() {
        return getString(ui.lineEditRestoreFile);
    }

    char GetAddHexOutput() {
        return ui.checkBoxOutputHex->isChecked();
    }
    
    char GetPrintAlgorithms() {
        return ui.checkBoxPrintAlgorithm->isChecked();
    }
    int GetTargetExecutionTimeMs() {
        return ui.spinBoxKernelStepTime->value();
    }
    int GetGpuBlocks() {
        return ui.spinBoxCUDANumberOfBlocks->value();
    }
    int GetGpuThreads() {
        return ui.spinBoxCUDANumberOfThreads->value();
    }
    /* TODO - iterate selected listWidgetDeviceSelection */
    std::vector<MFNDeviceInformation> GetDevicesToUse();

    char GetUseZeroCopy() {
        return ui.checkBoxCUDAZerocopy->isChecked();
    }
    char GetUseBfiInt() {
        return ui.checkBoxUseBfiPatching->isChecked();
    }
    int GetVectorWidth() {
        return ui.spinBoxVectorWidth->value();
    }


    char GetUseLookupTable() {
        return ui.checkBoxBigTableLookup->isChecked();
    }

    /* TODO */
    char GetVerbose() { return 0; }
    /* TODO */
    char GetSilent() { return 0; }
    /* TODO */
    char GetDaemon() { return 0; }
    /* TODO */
    char GetDebug() { return 0; }
    /* TODO */
    char GetDevDebug() { return 0; }

    int GetMinPasswordLength() {
        return ui.spinBoxMinPasswordLen->value();
    }
    int GetMaxPasswordLength() {
        return ui.spinBoxMaxPasswordLen->value();
    }


    // Returns zero if not set
    int GetWorkunitBits() {
        return ui.spinBoxWorkunitBits->value();
    }

    char GetIsNetworkServer() {
        return ui.checkBoxNetworkServerCompute->isChecked() ||
                ui.checkBoxNetworkServerNoCompute->isChecked();
    }

    char GetIsNetworkClient() {
        return getString(ui.lineEditNetworkRemoteHost).length() > 0 &&
                getString(ui.lineEditNetworkPort).length() > 0;
    }

    char GetIsServerOnly() {
        return ui.checkBoxNetworkServerNoCompute->isChecked();
    }

    std::string GetNetworkRemoteHostname() {
        return getString(ui.lineEditNetworkRemoteHost);
    }

    uint16_t GetNetworkPort() {
        return getInt(ui.lineEditNetworkPort);
    }

    /* TODO */
    std::vector<uint8_t> GetRestoreData(int passLength);

    /* TODO */
    void SetDataFromRestore(std::vector<uint8_t>);
    
private:
    std::string getString(QLineEdit *e) { return e->text().toStdString(); }
    int getInt(QLineEdit *e) {
        bool ok;
        int i = e->text().toInt(&ok);

        if(ok) return i;
        else return -1;
    }

    void populateDeviceList();

private:
    void connectPushButtonToLineEdit(QPushButton *button, QLineEdit *edit);

private slots:
    void pushButton_clicked(QWidget *edit);

private:
    Ui::MFNDisplayGuiSettings ui;

    // maps the push button's clicked() signal to pushButton_clicked with its
    // corresponding QLineEdit
    QSignalMapper *signalMapper;

    std::vector<MFNDeviceInformation> deviceList;
};

#endif // MAINWINDOW_H
