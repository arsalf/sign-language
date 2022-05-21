from numpy import sign
from signlanguage import SignLanguage
import os

sign = SignLanguage(os.path.join('MP_Data'), ['hello', 'thanks', 'iloveu'], 15, 30)
# sign.trainer_realtime('basic_sign.h5')
sign.realtime_test('basic_sign.h5')