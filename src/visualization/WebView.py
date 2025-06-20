import os
import tempfile
import plotly
from qtpy import QtWidgets
from qtpy.QtWebEngineWidgets import QWebEngineView
from qtpy.QtCore import QUrl

class WebView(QWebEngineView):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.plotly_fig = None
        self.html_file = None


    def update_fig(self, html):
        if self.html_file is not None: # close previous tempfiles so that they delete from disk properly
            self.html_file.close()
            os.unlink(self.html_file.name)

        self.html_file = tempfile.NamedTemporaryFile(suffix='.html', delete=False)
        with open(self.html_file.name, 'w') as file:
            file.write(html)
        url = QUrl.fromLocalFile(self.html_file.name)
        self.setUrl(url)

    def closeEvent(self, event):
        """Delete the temp file on close event."""
        super().closeEvent(event)
        if self.html_file is not None:
            self.html_file.close()
            os.unlink(self.html_file.name)
            self.html_file = None

    def __del__(self):
        """Handle temp file close (and deletion) if QWebEngineView is garbage collected instead of closed."""
        if self.html_file is not None:
            self.html_file.close()
            os.unlink(self.html_file.name)