from kivy.app import App
from kivy.uix.widget import Widget
import cv2  # opencv-python - requires numpy
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.screenmanager import ScreenManager, Screen
from  kivy.uix.popup import Popup
import numpy
import face_recognition
import os
from datetime import datetime

class RecognitionWidget(BoxLayout):
    pass


class RecognitionApp(App):
    screenManager = ScreenManager()
    showCam = True
    path = 'KnownFaces'
    images = []
    classNames = []
    unencodedFacesList = os.listdir(path)
    encodeListKnown = []
    namesToAddAfterSave = dict() # dict because fasta - it tells what to save after button press
    namesQueue = set() # queue for people to move into the namesToAddAfterSave
    namesAreQueued = False # do we need to schedule a move from namesQueue to namesToAddAfterSave?
    multiplyOfModelVideo = 1.0
    saveEveryTimeBool = False
    waitingForInput = True

    def build(self):
        #main screen
        saveButton = Button(text='Save')
        toggleCamButton = Button(text='toggle camera')
        settingsButton = Button(text='Settings')
        saveButton.size = 100, 100
        toggleCamButton.size = 100, 100
        settingsButton.size = 100, 100
        saveButton.bind(on_press=self.savefile)
        toggleCamButton.bind(on_press=self.toggleCam)
        settingsButton.bind(on_press=self.gotoSettings)

        self.img1 = Image()
        imageBox = BoxLayout()
        #imageBox.size = 640, 480
        #imageBox.size_hint = None, None
        buttonBox = BoxLayout()
        buttonBox.size_hint_y = None
        layoutMain = BoxLayout(orientation='vertical')
        self.layoutMainScreen = Screen()

        imageBox.add_widget(self.img1)
        buttonBox.add_widget(saveButton)
        buttonBox.add_widget(toggleCamButton)
        buttonBox.add_widget(settingsButton)
        layoutMain.add_widget(buttonBox)
        layoutMain.add_widget(imageBox)
        self.layoutMainScreen.add_widget(layoutMain)
        self.layoutMainScreen.name = 'layoutMainScreen'

        #settings screen
        mainGotoButton = Button(text='return')
        mainGotoButton.size = 100, 100
        mainGotoButton.size_hint_y = None
        mainGotoButton.bind(on_press=self.gotoMain)
        layoutSettings = BoxLayout(orientation='vertical')

        resizeBox = BoxLayout()
        resizeBox.size_hint_y = None
        resizeBox.size = 100, 100
        inpstr = 'Set multiply for model([0, 1]), current is {num}:'
        inpstr = inpstr.format(num=self.multiplyOfModelVideo)
        self.resizeLabel = Label(text=inpstr)
        self.resizeLabel.size_hint_y = None
        self.resizeLabel.size = 100, 30
        self.resizeTextInput = TextInput()
        self.resizeTextInput.multiline = False
        self.resizeTextInput.size = 30, 30
        self.resizeTextInput.size_hint_y = None
        self.resizeTextInput.hint_text = str(self.multiplyOfModelVideo)
        self.resizeTextInput.bind(on_text_validate=self.onEnterResizeTextInput)
        resizeBox.add_widget(self.resizeLabel)
        resizeBox.add_widget(self.resizeTextInput)

        clearDictBox = BoxLayout()
        clearDictLabel = Label(text='Mark attendance after each save')
        clearDictCheckBox = CheckBox()
        clearDictCheckBox.size = 100, 30
        clearDictCheckBox.bind(active=self.saveEveryTime)
        clearDictCheckBox.size_hint_y = None
        clearDictLabel.size_hint_y = None
        clearDictLabel.size = 100, 30
        clearDictBox.size_hint_y = None
        clearDictBox.size = 100, 100
        clearDictBox.add_widget(clearDictLabel)
        clearDictBox.add_widget(clearDictCheckBox)


        self.layoutSettingsScreen = Screen()

        layoutSettings.add_widget(clearDictBox)
        layoutSettings.add_widget(resizeBox)
        layoutSettings.add_widget(mainGotoButton)
        self.layoutSettingsScreen.add_widget(layoutSettings)
        self.layoutSettingsScreen.name = 'layoutSettingsScreen'

        #empty screen
        self.layoutEmptyScreen = Screen()
        self.layoutEmptyScreen.name = "layoutEmptyScreen"
        # fin
        self.screenManager.add_widget(self.layoutMainScreen)
        self.screenManager.add_widget(self.layoutSettingsScreen)
        self.screenManager.add_widget(self.layoutEmptyScreen)

        # opencv2 stuffs
        self.capture = cv2.VideoCapture(0)  # get the first camera
        success, frame = self.capture.read()
        if not success:
            self.errPopup("Failed to get video from camera", "retry", self.retryLoadCamera)
        else:
            # get faces to read
            for cls in self.unencodedFacesList:
                curImg = cv2.imread(f'{self.path}/{cls}')
                self.images.append(curImg)
                self.classNames.append(os.path.splitext(cls)[0])
            self.encodeListKnown = self.findEncodings(self.images)
            Clock.schedule_interval(self.update, 1.0 / 33.0)

        return self.screenManager
        # return RecognitionWidget()

    def update(self, dt):
        success, frame = self.capture.read()  # get the frame
        rgb_frame = frame[:, :, ::-1]  # BGR2RGB
        if self.multiplyOfModelVideo != 1.0:
            rgb_frame = cv2.resize(frame, (0, 0), None, self.multiplyOfModelVideo, self.multiplyOfModelVideo) # could do that to make it faster
        face_locations = face_recognition.face_locations(rgb_frame)  # find faces
        encoded_faces = face_recognition.face_encodings(rgb_frame, face_locations)

        for encodeFace, faceLoc in zip(encoded_faces, face_locations): # zip to only go through relevant pairs, not all possible ones
            matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
            matchIndex = numpy.argmin(faceDis)  # faster

            if matches[matchIndex]:
                name = self.classNames[matchIndex]
                #self.markAttendance(name) # this suxxx performance a lot
                self.namesQueue.add(name) # this shouldn't take much time
                if not self.namesAreQueued:
                    self.namesAreQueued = True
                    Clock.schedule_once(self.emptyNamesQueue, 10) # every 10 seconds we update the list
                if self.showCam:
                    top, right, bottom, left = faceLoc
                    top *= int(1 / self.multiplyOfModelVideo)
                    right *= int(1 / self.multiplyOfModelVideo)
                    bottom *= int(1 / self.multiplyOfModelVideo)
                    left *= int(1 / self.multiplyOfModelVideo)
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # if (self. showCam): # draw a rectangle around the face
            #    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # show the camera footage
        if self.showCam:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)  # this should have the border around the face, but doesn't, ok
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            # if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.img1.texture = texture1

    def toggleCam(instance, value):
        instance.showCam = not instance.showCam
        instance.img1.texture = None
        pass

    def findEncodings(self, images):
        encodeList = []
        failures = []
        for i in range(len(images)):
            try:
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(images[i])[0]
                encodeList.append(encode)
            except:
                failures.append(self.classNames[i]) # bruh?
        if len(failures) != 0:
            err = "Failed to recognise those images: "
            for f in failures:
                err += f + " "
            self.errPopup(err, "ok")
        return encodeList

    def savefile(self, instance): # so basicly we have a recent queue from which we push to the long-time queue, and the to the file when the save button is pressed
        self.emptyNamesQueue(0)
        for name in self.namesToAddAfterSave.keys():
            with open("Attendance.csv", "r+") as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0]) # btw checking every time is not cool performance-wise, but not critical
                if self.saveEveryTimeBool or name not in nameList: # commented cause we check it in the names to add dict
                    f.writelines(f'\n{name}, {self.namesToAddAfterSave[name]}')
        self.namesToAddAfterSave.clear() # not sure if we don't need that
    def emptyNamesQueue(self, dt):
        for name in self.namesQueue:
            self.namesToAddAfterSave[name] = datetime.now()
            print(name)
        self.namesQueue.clear()
        self.namesAreQueued = False
    def gotoSettings(self, instance):
        self.screenManager.current = 'layoutSettingsScreen'

    def gotoMain(self, instance = None):
        self.screenManager.current = 'layoutMainScreen'

    def onEnterResizeTextInput(self, v):
        value = float(self.resizeTextInput.text)
        self.multiplyOfModelVideo = numpy.clip(value, 0.001, 1)
        inpstr = 'Set multiply for model([0, 1]), current is {num}:'
        inpstr = inpstr.format(num=self.multiplyOfModelVideo)
        self.resizeLabel.text = inpstr

    def saveEveryTime(self, somethingidunno, value):
        if value:
            self.saveEveryTimeBool = True
        else:
            self.saveEveryTimeBool = False

    def retryLoadCamera(self, caller = None):
        #self.capture = cv2.VideoCapture(0)
        #success, frame = self.capture.read()
        #self.gotoMain(self)
        #if not success: self.errPopup("Failed to get video from camera", "retry")
        #else: Clock.schedule_interval(self.update, 1.0 / 33.0)
        exit(0) # bruh
    def errPopup(self, errtext, buttontext, callback = None):
        self.screenManager.current = 'layoutEmptyScreen' # for some reason a pop up doesn't actually pop up - we need to place it on empty screen
        popBox = BoxLayout(orientation="vertical")
        l = Label(text=errtext)
        b = Button(text=buttontext)
        popBox.add_widget(l)
        popBox.add_widget(b)
        popup = Popup(title='Error',
                      content=popBox,
                      size_hint=(None, None), size=(400, 400),
                      auto_dismiss=False)

        if callback:
            b.on_touch_up = callback
            b.on_press = popup.dismiss
        else: # you probably shouldn't do that
            b.on_touch_up = self.gotoMain
            b.on_press = popup.dismiss
        popup.open()






if __name__ == '__main__':
    RecognitionApp().run()
    cv2.destroyAllWindows()

'''
import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
'''
