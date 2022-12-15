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
import numpy
import face_recognition
import os
from datetime import datetime
class RecognitionWidget(BoxLayout):
    pass


class RecognitionApp(App):
    showCam = True
    path = 'KnownFaces'
    images = []
    classNames = []
    unencodedFacesList = os.listdir(path)
    encodeListKnown = []
    namesToAddAfterSave = dict() # dict because fasta - it tells what to save after button press
    namesQueue = set() # queue for people to move into the namesToAddAfterSave
    namesAreQueued = False # do we need to schedule a move from namesQueue to namesToAddAfterSave?

    def build(self):
        self.img1 = Image()
        btn1 = Button(text='Save')
        btn2 = Button(text='toggle camera')
        imageBox = BoxLayout()
        #imageBox.size = 640, 480
        #imageBox.size_hint = None, None
        buttonBox = BoxLayout()
        buttonBox.size_hint_y = None
        layout = BoxLayout(orientation='vertical')
        btn1.size = 100, 100
        btn2.size = 100, 100
        btn1.bind(on_press=self.savefile)
        btn2.bind(on_press=self.toggleCam)
        imageBox.add_widget(self.img1)
        buttonBox.add_widget(btn1)
        buttonBox.add_widget(btn2)
        layout.add_widget(buttonBox)
        layout.add_widget(imageBox)


        # get faces to read
        for cls in self.unencodedFacesList:
            curImg = cv2.imread(f'{self.path}/{cls}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cls)[0])
        self.encodeListKnown = self.findEncodings(self.images)
        # opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout
        # return RecognitionWidget()

    def update(self, dt):
        success, frame = self.capture.read()  # get the frame
        rgb_frame = frame[:, :, ::-1]  # BGR2RGB
        rgb_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25) # could do that to make it faster
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
                    for top, right, bottom, left in face_locations:
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left + 6, bottom - 6),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),2)

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
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def savefile(self, instance): # so basicly we have a recent queue from which we push to the long-time queue, and the to the file when the save button is pressed
        self.emptyNamesQueue(0)
        for name in self.namesToAddAfterSave.keys():
            with open("Attendance.csv", "r+") as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                if name not in nameList:
                    f.writelines(f'\n{name}, {self.namesToAddAfterSave[name]}')
        #self.namesToAddAfterSave.clear() # not sure if we need that
    def emptyNamesQueue(self, dt):
        for name in self.namesQueue:
            self.namesToAddAfterSave[name] = datetime.now()
            print(name)
        self.namesQueue.clear()
        self.namesAreQueued = False



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
