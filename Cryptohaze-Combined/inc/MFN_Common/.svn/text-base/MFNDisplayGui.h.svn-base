/*
Cryptohaze Multiforcer & Wordyforcer - low performance GPU password cracking
Copyright (C) 2011  Bitweasil (http://www.cryptohaze.com/)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
*/

#ifndef __MFNDISPLAY_GUI_H
#define __MFNDISPLAY_GUI_H

#include "MFN_Common/MFNDisplay.h"
#include "MFN_Common/MFNDisplayGuiSettings.h"
#include "MFN_Common/MFNWorkunitBase.h"
#include "CH_Common/CHHiresTimer.h"
#include "CH_HashFiles/CHHashFileV.h"

#include <QMainWindow>
#include <QSystemTrayIcon>
#include <QLabel>
#include <QPushButton>

#include <string>
#include <vector>

#include <stdint.h>

#include "ui_MFNDisplayGui.h"

class CHHashFileV;
class MFNWorkunitBase;

class MFNDisplayGui : public QMainWindow, public MFNDisplay {
    Q_OBJECT

public:
    MFNDisplayGui(QWidget *parent = 0);

public:

    virtual void Refresh();
    
    /**
     * Sets the currently-active hash name.
     * @param newHashName String of the hash name.
     */
    virtual void setHashName(std::string newHashName);

    /**
     * Sets the current password length being cracked.
     * @param newPasswordLength New length.
     */
    virtual void setPasswordLen(uint16_t newPasswordLength);
      
    /**
     * Adds a new cracked password.  This is a vector, as that is how the
     * passwords are handled internally.
     * @param newFoundPassword A vector containing the password string.
     */
    virtual void addCrackedPassword(std::vector<uint8_t> newFoundPassword);

    /**
     * Adds a new status line to the system.
     * @param newStatusLine std::string or char* status line.
     */
    virtual void addStatusLine(std::string newStatusLine);
    virtual void addStatusLine(char * newStatusLine);

    // Sets the system mode: Standalone, network server, network client.
    virtual void setSystemMode(int systemMode, std::string modeString);

    // Add or subtract from the number of connected clients
    virtual void alterNetworkClientCount(int networkClientCount);

private:
    void createControlButtonbox();
    void createTray();
    void createActions();

    void showMinimizedBalloon();

    void checkIdleTime();

private slots:
    void showSettingsClicked();

    void startButtonClicked();
    void exitApp();

    void trayActivated(QSystemTrayIcon::ActivationReason reason);
    void closeEvent(QCloseEvent *event);

private:
    Ui::MFNDisplayGui ui;
    MFNDisplayGuiSettings settings;

    bool shownMinimizedBalloon;
    QAction *showAction;
    QAction *minimizeAction;
    QAction *showSettingsAction;
    QAction *restoreAction;
    QAction *quitAction;

    QSystemTrayIcon *trayIcon;
    QMenu *trayIconMenu;

    QLabel *labelWorkUnits;
    QLabel *labelNetClients;

    QPushButton *pushButtonStart;

private:
    bool isRunning;

    int systemMode;
    int numNetClients;


    CHHiresTimer displayTimer;
    double lastTimeUpdated;

    uint32_t workunitsCompleted;
    uint32_t workunitsTotal;
    uint64_t crackedHashes;
    uint64_t totalHashes;


};


#endif
