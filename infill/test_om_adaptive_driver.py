
import sys
import unittest
from math import sin
from io import StringIO

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_partials
from openmdao.utils.logger_utils import TestLogger