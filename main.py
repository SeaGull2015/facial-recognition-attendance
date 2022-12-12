from kivy.app import App
from kivy.uix.widget import Widget
import cv2 # opencv-python - requires numpy
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import face_recognition

class RecognitionWidget(BoxLayout):
    pass
    

class RecognitionApp(App):
    showCam = False

    def savefile(instance):
        pass

    def toggleCam(instance, value):
        instance.showCam = not instance.showCam
        instance.img1.texture = None
        pass
    def build(self):
        self.img1 = Image()
        btn1 = Button(text='Save to')
        btn2 = Button(text='toggle camera')
        layout = BoxLayout(orientation='vertical')
        btn1.size = 100, 100
        btn2.size = 100, 100
        btn1.size_hint = None, None
        btn2.size_hint = None, None
        btn1.bind(on_press=self.savefile)
        btn2.bind(on_press=self.toggleCam)
        layout.add_widget(btn1)
        layout.add_widget(btn2)
        layout.add_widget(self.img1)

        # opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout
        #return RecognitionWidget()

    def update(self, dt):
        # display image from cam in opencv window
        ret, frame = self.capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        if (self.showCam):
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # convert it to texture
            buf1 = cv2.flip(frame, 0) # this should have the border around the face, but doesn't, ok
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            # if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.img1.texture = texture1




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