#!/usr/bin/python
activate_this = '/var/www/flask_ssd/flask_ssd/keras1_tf1/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/flask_ssd/")
from flask_ssd import app as application
