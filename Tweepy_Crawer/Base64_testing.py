import base64
import hashlib
user_handler = 'KP_Taipei'
#Base64
B64_user = base64.b64decode(user_handler)
#print(str(en_user).strip("b'"))
print(base64.b64encode(B64_user))

#MD5
MD5_user = hashlib.md5(B64_user).hexdigest()
print(MD5_user)

