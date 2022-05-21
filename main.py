from numpy import sign
from signlanguage import SignLanguage
import os

sign = SignLanguage(os.path.join('MP_Data'), ['hello', 'thanks', 'iloveu'], 15, 30)
sign.realtime_trainer()
sign.realtime_test('hello-thanks-iloveu.h5')
print(sign.file_name)