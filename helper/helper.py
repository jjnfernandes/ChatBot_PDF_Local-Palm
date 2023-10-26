import os
import toml

class Helper:
    def createFile(self, file):
        fileLocation = f"temp/{file.name}"
        with open(fileLocation, "wb+") as fileObject:
            fileObject.write(file.getbuffer())

        return fileLocation

    def deleteFile(self, filePath):
        os.remove(filePath)

    def set_api_key(self,google_api_key):
        with open(".streamlit/secrets.toml", "r+") as f:
            secrets = toml.load(f)
            secrets["palm_api_key"] = google_api_key
            f.seek(0)
            toml.dump(secrets, f)
            f.truncate()
        return True