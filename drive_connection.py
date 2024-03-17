from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Authenticate and create a GoogleDrive instance
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # This will open a new tab in your web browser for authentication
drive = GoogleDrive(gauth)