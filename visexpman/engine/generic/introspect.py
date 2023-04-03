import logging,platform
import logging.handlers
log = logging.getLogger('introspect')
try:
    import PyQt4.QtCore as QtCore
except ImportError:
    import PyQt5.QtCore as QtCore
from contextlib import contextmanager
import inspect
import time
import sys
import re
import numpy
import hashlib
## {{{ http://code.activestate.com/recipes/519621/ (r4)
import weakref

import subprocess, os, signal
import numpy
import psutil

def visexpman2hash():
    from visexpman.engine.generic import fileop
    foldername=fileop.visexpman_package_path()
    files=[]
    for subfold in [os.path.join('users','common'), 'engine']:
        files.extend(fileop.find_files_and_folders(os.path.join(foldername, subfold))[1])
    sha=hashlib.sha256()
    files=[sha.update(fileop.read_text_file(f)) for f in files if os.path.splitext(f)[1]=='.py']
    return numpy.fromstring(sha.digest(), dtype=numpy.uint8)
    
def mes2hash():
    if platform.system()=='Windows':
        from visexpman.engine.generic import fileop
        folder='C:\\MES\\MES5'
        sha=hashlib.sha256()
        [sha.update(fileop.read_text_file(f)) for f in fileop.find_files_and_folders(folder, extension='mat')[1]]
        return numpy.fromstring(sha.digest(), dtype=numpy.uint8)
    else:
        return numpy.array([], dtype=numpy.uint8)

def cap_attributes2dict(obj):
    '''
    All attributes which are capitalized are returned as a dict
    '''
    return dict([(vn, getattr(obj, vn)) for vn in dir(obj) if vn.isupper()])

def get_available_process_cores():
    '''
    Returns the number of available processor cores. It is the 75% of all the cores but at least 1
    '''
    import multiprocessing
    multiprocessing.cpu_count()
    nprocesses = int(multiprocessing.cpu_count()*0.75)
    return 1 if nprocesses==0 else nprocesses

def get_process_name(pid):
    p = psutil.Process(pid)
    if callable(p.name):
        return p.name()
    else:
        return p.name

def get_python_processes():
    pids = []
    for pid in (psutil.pids() if hasattr(psutil, 'pids') else psutil.get_pid_list()):
        try:
            if 'python' in get_process_name(pid):
                pids.append(pid)
        except:
            pass
    return pids
    
def kill_python_processes(dont_kill_pids):
    killed = []
    pids = get_python_processes()
    for pid in pids:
        if pid not in dont_kill_pids:
            killed.append(pid)
            p = psutil.Process(pid)
            p.kill()
    return killed
    
def kill_other_python_processes():
    kill_python_processes([os.getpid()])
    
def python_memory_usage():
    res=[]
    for pid in get_python_processes():
        p=psutil.Process(pid)
        res.append([pid, p.memory_info()])
    return res

def dumpall(fn):
    '''
    Can be used when exception occurs to save the whole context of the software
    '''
    f = open(fn,'wt')
    import gc
    errct = 0
    for o in gc.get_objects():
        try:
            ignore_types = [int,float,tuple,list,dict]
            ignore_classes = ['type', 'module', 'function']
            if any(map(isinstance, len(ignore_types)*[o], ignore_types)) or o == [] or o == {}:
                continue
            if hasattr(o, '__class__') and hasattr(o.__class__, '__name__') and o.__class__.__name__ in ignore_classes:
                continue
            if 'visexp' in str(o):
                f.write(object2str(o) + '\r\n\r\n=================================================\r\n')
        except:
            errct += 1
    print( errct)
    f.close()
    

def object2str(object):
    object_variable_values = 'Classname: {0}, id: 0x{1:x}'.format(object.__class__.__name__, id(object))
    for vn in dir(object):
        if vn[:2] != '__' and vn[-2:] != '__':
            object_variable_values = object_variable_values + '\r\n{0}={1}'.format(vn, getattr(object, vn))
    return object_variable_values

def is_test_running():
    '''
    Finds out if test is being run: either unittest.main() or unittest_aggregator module is in call stack
    '''
    if hasattr(is_test_running,'isrunning'):
        return is_test_running.isrunning
    import traceback
    keywords = ['unittest.main()', 'unittest_aggregator']
    isrunning = False
    if '--unittest' in sys.argv and  '-c' in sys.argv or '--testmode' in sys.argv:#When called as python -c code --unittest. This is used for testing qapps
        isrunning = True
    for item in traceback.format_stack():
        for keyword in keywords:
            if keyword in item:
                isrunning = True
    is_test_running.isrunning = isrunning
    return isrunning

def class_ancestors(obj):
    ancestors = []
    if not hasattr(obj, '__bases__'):
        return ancestors
    ancestor = [obj]
    while True:
        ancestor = map(getattr, list(ancestor), len(ancestor)*['__bases__'])
        flattened_ancestors = []
        for a in ancestor:
            for ai in a:
                flattened_ancestors.append(ai)
        ancestor = flattened_ancestors
        ancestors.extend(map(getattr, list(flattened_ancestors), len(flattened_ancestors)*['__name__']))
        if 'object' in ancestors:
            break
    return ancestors
    

def import_non_local(name, custom_name=None, exclude_string=None):
    import imp
    custom_name = custom_name or name
    if exclude_string is None:
        paths = sys.path[1:]
    else:
        paths = [p1 for p1 in sys.path[1:] if exclude_string not in p1]
    f, pathname, desc = imp.find_module(name, paths)
    module = imp.load_module(custom_name, f, pathname, desc)
    if hasattr(f,'close'):
        f.close()
    return module

def kill_child_processes(parent_pid, sig='SIGTERM'):
        ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % parent_pid, shell=True, stdout=subprocess.PIPE)
        ps_output = ps_command.stdout.read()
        retcode = ps_command.wait()
        assert retcode == 0, "ps command returned %d" % retcode
        for pid_str in ps_output.split("\n")[:-1]:
            try:
                os.kill(int(pid_str), getattr(signal, sig))
            except:
                pass


# Importing a dynamically generated module

def import_code(code,name,add_to_sys_modules=0):
    """
    Import dynamically generated code as a module. code is the
    object containing the code (a string, a file handle or an
    actual compiled code object, same types as accepted by an
    exec statement). The name is the name to give to the module,
    and the final argument says wheter to add it to sys.modules
    or not. If it is added, a subsequent import statement using
    name will return this module. If it is not added to sys.modules
    import will try to load it in the normal fashion.

    import foo

    is equivalent to

    foofile = open("/path/to/foo.py")
    foo = importCode(foofile,"foo",1)

    Returns a newly generated module.
    """
    import sys,imp

    module = imp.new_module(name)

    exec (code in module.__dict__)
    if add_to_sys_modules:
        sys.modules[name] = module
    return module

def string2objectreference(self, reference_string):
    items = reference_string.split('.')
    if items[0] == 'self':
        reference = getattr(self, items[1])
        for item in items[2:]:
            if '[' in item:
                reference = getattr(reference, item.split('[')[0])
                if '\'' in item:
                    index_key = item.split('[')[1].replace(']',  '').replace('\'','') 
                else:
                    index_key = int(item.split('[')[1].replace(']',''))
                reference = reference[index_key]
            else:
                reference = getattr(reference, item)
        return reference
    else:
        return None
    
class VerifyInstallation(object):
    def __init__(self):
        self.system=platform.system()
        self.verify_modules()
        self.verify_pygame()
        self.verify_qt()
        self.verify_serial()
        self.verify_paramiko()
        
    def verify_modules(self):
        expected_modules=['pygame', 'OpenGL', 'pyqtgraph', 'PyDAQmx', 'visexpman', 'zc.lockfile', 
                    'serial', 'cv2', 'hdf5io', 'tables']
        missing_modules=[]
        for em in expected_modules:
            try:
                __import__(em)
            except:
                missing_modules.append(em)
        if len(missing_modules)>0:
            raise RuntimeError('Module(s) not installed: {0}'.format(', '.join(missing_modules)))
        
    def verify_serial(self):
        import serial
        if self.system=='Linux':
            port='/dev/ttyUSB0'
            return
        elif self.system=='Windows':
            port='COM1'
        s=serial.Serial(port,timeout=1)
        s.write('test')
        time.sleep(0.3)
        s.close()
        
    def verify_paramiko(self):
        import fileop,tempfile
        path='v:\\' if self.system=='Windows' else '/mnt/datafast'
        pw=fileop.read_text_file(os.path.join(path, 'codes','jobhandler','pw.txt')).title()
        fileop.download_folder('rldata.fmi.ch', 'mouse', '/data/software/rldata/visexpman', tempfile.gettempdir(), password=pw)
        
    def verify_pygame(self):
        from visexpman.engine.visexp_app import stimulation_tester
        stimulation_tester('zoltan', 'StimulusDevelopment', 'ShortTestStimulus')

    def verify_qt(self):
        from visexpman.engine.generic import gui
        gui=gui.SimpleAppWindow()
  
import unittest
class TestUtils(unittest.TestCase):
    def setUp(self):
        pass
        
    def tearDown(self):
        pass
        
    def test_01_folder2hash(self):
        visexpman2hash()

    def test_02_ModifiableIterator(self):
        list = [1,2,3,4]
        alist = ModifiableIterator(list)
        result=[]
        for item in alist:
            if item==2:
                alist.list = [1,3,4,5]
            result.append(item)
        self.assertEqual(result,[1,2,1,3,4,5])
        
    def test_03_installation_tester(self):
        VerifyInstallation()
        
        
        
        pass
        
    def test_flatten(self):
        a = []
        for i in xrange(3):
            a = [a, i]
            a = flatten(a)
        #self.assertEqual()
        
    def test_dynamic_import(self):
        # Example
        code = \
        """
        def testFunc():
            print "spam!"

        class testClass:
            def testMethod(self):
                print "eggs!"
        """

        m = import_code(code,"test")
        m.testFunc()
        o = m.testClass()
        o.testMethod()
        ## end of http://code.activestate.com/recipes/82234/ }}}
    
if __name__=='__main__':
    unittest.main()
   






    
