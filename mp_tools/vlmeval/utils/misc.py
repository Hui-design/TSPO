import os
import datetime
import pandas as pd
import os.path as osp
import logging
from sty import fg, bg, ef, rs


def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip = re.compile('(\d)(,)(\d)')  # noqa: W605
    periodStrip = re.compile('(?!<=\d)(\.)(?!\d)')  # noqa: W605
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def h2r(value):
    if value[0] == '#':
        value = value[1:]
    assert len(value) == 6
    return tuple(int(value[i:i + 2], 16) for i in range(0, 6, 2))

def r2h(rgb):
    return '#%02x%02x%02x' % rgb

def colored(s, color):
    if isinstance(color, str):
        if hasattr(fg, color):
            return getattr(fg, color) + s + fg.rs
        color = h2r(color)
    return fg(*color) + s + fg.rs

def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(eval(s), type)
    except Exception as _:
        return False

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def d2df(D):
    return pd.DataFrame({x: [D[x]] for x in D})

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False

def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError

def load_env():
    logger = logging.getLogger('LOAD_ENV')
    try:
        import vlmeval
    except ImportError:
        logger.error('TeleVLMEval is not installed. Failed to import environment variables from .env file. ')
        return
    pth = osp.realpath(vlmeval.__path__[0])
    pth = osp.join(pth, '../.env')
    pth = osp.realpath(pth)
    if not osp.exists(pth):
        logger.error(f'Did not detect the .env file at {pth}, failed to load. ')
        return

    from dotenv import dotenv_values
    values = dotenv_values(pth)
    for k, v in values.items():
        if v is not None and len(v):
            os.environ[k] = v
    logger.info(f'API Keys successfully loaded from {pth}')

def get_available_gpus(max_mem_used=80*1024):
    import GPUtil
    gpus = GPUtil.getGPUs()
    available_gpus = [gpu.id for gpu in gpus if gpu.memoryUsed < max_mem_used]
    return available_gpus

def timestr(second=True, minute=False):
    s = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:]
    if second:
        return s
    elif minute:
        return s[:-2]
    else:
        return s[:-4]
