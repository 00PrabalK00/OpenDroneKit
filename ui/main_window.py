"""Main window for the finalized structural inspection toolkit."""

from PyQt6.QtWidgets import QLabel, QMainWindow, QStatusBar, QTabWidget, QVBoxLayout, QWidget

from .workspace import (
    AppSession,
    DataTab,
    FlyTab,
    MissionsWorkspaceTab,
    ProjectsTab,
    ReportsTab,
    SettingsTab,
)
from .tabs import CrackAnalysisTab, FullPipelineTab, MetalDefectTab, OperationsCenterTab, ReconstructionTab
from .theme import standard_icon


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenDroneKit")
        self.resize(1480, 900)

        self.session = AppSession()

        tabs = QTabWidget(self)
        tabs.setDocumentMode(True)
        tabs.setMovable(False)

        projects_tab = ProjectsTab(self.session)
        missions_tab = MissionsWorkspaceTab(self.session)
        fly_tab = FlyTab(self.session)
        data_tab = DataTab(self.session)
        reports_tab = ReportsTab(self.session)
        settings_tab = SettingsTab(self.session)

        tabs.addTab(projects_tab, standard_icon(self, "SP_DirHomeIcon", "SP_DirIcon"), "Projects")
        tabs.addTab(
            missions_tab,
            standard_icon(self, "SP_FileDialogDetailedView", "SP_FileDialogListView"),
            "Missions",
        )
        tabs.addTab(fly_tab, standard_icon(self, "SP_MediaPlay", "SP_ArrowForward"), "Fly")
        tabs.addTab(data_tab, standard_icon(self, "SP_DriveHDIcon", "SP_FileDialogInfoView"), "Data")
        tabs.addTab(reports_tab, standard_icon(self, "SP_FileIcon", "SP_DialogSaveButton"), "Reports")
        tabs.addTab(
            settings_tab,
            standard_icon(self, "SP_FileDialogContentsView", "SP_TitleBarMenuButton"),
            "Settings",
        )

        analysis_host = QWidget()
        analysis_layout = QVBoxLayout(analysis_host)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_tabs = QTabWidget(analysis_host)
        analysis_tabs.setDocumentMode(True)
        analysis_tabs.addTab(
            OperationsCenterTab(self.session),
            standard_icon(self, "SP_ComputerIcon", "SP_DesktopIcon"),
            "Operations",
        )
        analysis_tabs.addTab(
            FullPipelineTab(self.session),
            standard_icon(self, "SP_BrowserReload", "SP_ArrowRight"),
            "Full Pipeline",
        )
        analysis_tabs.addTab(
            CrackAnalysisTab(self.session),
            standard_icon(self, "SP_MessageBoxWarning", "SP_BrowserStop"),
            "Crack + Propagation",
        )
        analysis_tabs.addTab(
            ReconstructionTab(self.session),
            standard_icon(self, "SP_DialogOpenButton", "SP_DirOpenIcon"),
            "3D Reconstruction",
        )
        analysis_tabs.addTab(
            MetalDefectTab(self.session),
            standard_icon(self, "SP_DialogApplyButton", "SP_DialogYesButton"),
            "Defect Models",
        )
        analysis_layout.addWidget(analysis_tabs)
        tabs.addTab(
            analysis_host,
            standard_icon(self, "SP_FileDialogInfoView", "SP_MessageBoxInformation"),
            "Analysis",
        )
        tabs.setObjectName("topTabs")
        self.setCentralWidget(tabs)

        status = QStatusBar(self)
        self.setStatusBar(status)
        self.system_badge = QLabel("System Online")
        self.system_badge.setStyleSheet("QLabel { color: #35c77f; font-weight: 700; padding-right: 8px; }")
        status.addPermanentWidget(self.system_badge)
        self.session.statusChanged.connect(status.showMessage)
        status.showMessage("Workspace ready")

    def closeEvent(self, event):
        try:
            self.session.store.close()
        except Exception:
            pass
        super().closeEvent(event)
