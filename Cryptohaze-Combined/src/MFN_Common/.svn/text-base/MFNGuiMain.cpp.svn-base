#include <QtGui>

#include "MFN_Common/MFNDisplayGui.h"

#include "MFN_Common/MFNRun.h"
#include "MFN_Common/MFNCommandLineData.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    global_interface.exit = 0;
    global_interface.user_exit = 0;
    global_interface.pause = 0;

    if (!QSystemTrayIcon::isSystemTrayAvailable()) {
        QMessageBox::critical(0, QObject::tr("Systray"),
                              QObject::tr("I couldn't detect any system tray "
                                          "on this system."));
        return 1;
    }
    QApplication::setQuitOnLastWindowClosed(false);

    MFNDisplayGui window;
    //MFNDisplayGuiSettings window;
    window.show();


    return app.exec();
}

#ifdef WIN32

#include <Windows.h>
/* If no win32 is specified when adding exectuable in cmake, it uses cmd.exe 
    to execute main().  This means we get a cmd.exe prompt that starts our gui.
    To avoid it, using ADD_EXECUTABLE( blah WIN32 ... ) which relies on the 
    WinMain entrypoint */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR szCmdLine, int iCmdShow)
{
    return main(0, NULL);
}

#endif
