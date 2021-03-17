import os, shutil, glob
import numpy as np
from typing import List, Mapping, Union
from base_test import ArkoudaTest
from context import arkouda as ak
from arkouda import io_util
import h5py

'''
Tests writing Arkouda pdarrays to and reading from files
'''
class IOTest(ArkoudaTest):

    @classmethod
    def setUpClass(cls):
        super(IOTest, cls).setUpClass()
        IOTest.io_test_dir = '{}/io_test'.format(os.getcwd())
        io_util.get_directory(IOTest.io_test_dir)

    def setUp(self):
        ArkoudaTest.setUp(self)
        self.int_tens_pdarray = ak.array(np.random.randint(-100,100,1000))
        self.int_tens_ndarray = self.int_tens_pdarray.to_ndarray()
        self.int_tens_ndarray.sort()
        self.int_tens_pdarray_dupe = ak.array(np.random.randint(-100,100,1000))

        self.int_hundreds_pdarray = ak.array(np.random.randint(-1000,1000,1000))
        self.int_hundreds_ndarray = self.int_hundreds_pdarray.to_ndarray()
        self.int_hundreds_ndarray.sort()
        self.int_hundreds_pdarray_dupe = ak.array(np.random.randint(-1000,1000,1000))

        self.float_pdarray = ak.array(np.random.uniform(-100,100,1000))  
        self.float_ndarray = self.float_pdarray.to_ndarray() 
        self.float_ndarray.sort()
        self.float_pdarray_dupe = ak.array(np.random.uniform(-100,100,1000))   

        self.bool_pdarray = ak.randint(0, 1, 1000, dtype=ak.bool)
        self.bool_pdarray_dupe = ak.randint(0, 1, 1000, dtype=ak.bool)     
   
        self.dict_columns =  {
           'int_tens_pdarray' : self.int_tens_pdarray,
           'int_hundreds_pdarray' : self.int_hundreds_pdarray,
           'float_pdarray' : self.float_pdarray,
           'bool_pdarray' : self.bool_pdarray
        }
        
        self.dict_columns_dupe =  {
           'int_tens_pdarray' : self.int_tens_pdarray_dupe,
           'int_hundreds_pdarray' : self.int_hundreds_pdarray_dupe,
           'float_pdarray' : self.float_pdarray_dupe,
           'bool_pdarray' : self.bool_pdarray_dupe
        }
        
        self.dict_single_column = {
           'int_tens_pdarray' : self.int_tens_pdarray
        }
        
        self.list_columns = [
          self.int_tens_pdarray,
          self.int_hundreds_pdarray,
          self.float_pdarray,
          self.bool_pdarray
        ]
        
        self.names =  [
          'int_tens_pdarray',
          'int_hundreds_pdarray',
          'float_pdarray',
          'bool_pdarray'
        ]

    def _create_file(self, prefix_path : str, columns : Union[Mapping[str,ak.array]], 
                                           names : List[str]=None) -> None:
        '''
        Creates an hdf5 file with dataset(s) from the specified columns and path prefix
        via the ak.save_all method. If columns is a List, then the names list is used 
        to create the datasets
        
        :return: None
        :raise: ValueError if the names list is None when columns is a list
        '''       
        if isinstance(columns, dict):
            ak.save_all(columns=columns, prefix_path=prefix_path)   
        else:
            if not names:
                raise ValueError('the names list must be not None if columns is a list')
            ak.save_all(columns=columns, prefix_path=prefix_path, names=names)
    
    def testSaveAllLoadAllWithDict(self): 

        '''
        Creates 2..n files from an input columns dict depending upon the number of 
        arkouda_server locales, retrieves all datasets and correspoding pdarrays, 
        and confirms they match inputs
        
        :return: None
        :raise: AssertionError if the input and returned datasets and pdarrays don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict'.format(IOTest.io_test_dir))
        retrieved_columns = ak.load_all('{}/iotest_dict'.format(IOTest.io_test_dir))

        itp = self.dict_columns['int_tens_pdarray'].to_ndarray()
        ritp = retrieved_columns['int_tens_pdarray'].to_ndarray()
        itp.sort()
        ritp.sort()
        ihp = self.dict_columns['int_hundreds_pdarray'].to_ndarray()
        rihp = retrieved_columns['int_hundreds_pdarray'].to_ndarray()
        ihp.sort()
        rihp.sort()
        ifp = self.dict_columns['float_pdarray'].to_ndarray()
        rifp = retrieved_columns['float_pdarray'].to_ndarray()
        ifp.sort()
        rifp.sort()

        self.assertEqual(4, len(retrieved_columns))
        self.assertTrue((itp == ritp).all())
        self.assertTrue((ihp == rihp).all())
        self.assertTrue((ifp == rifp).all())    
        self.assertEqual(len(self.dict_columns['bool_pdarray']),  
                         len(retrieved_columns['bool_pdarray']))    
        self.assertEqual(4, 
                         len(ak.get_datasets('{}/iotest_dict_LOCALE0000'.format(IOTest.io_test_dir))))
        
    def testSaveAllLoadAllWithList(self):
        '''
        Creates 2..n files from an input columns and names list depending upon the number of 
        arkouda_server locales, retrieves all datasets and correspoding pdarrays, and confirms 
        they match inputs
        
        :return: None
        :raise: AssertionError if the input and returned datasets and pdarrays don't match
        '''
        self._create_file(columns=self.list_columns, 
                          prefix_path='{}/iotest_list'.format(IOTest.io_test_dir), 
                          names=self.names)
        retrieved_columns = ak.load_all(path_prefix='{}/iotest_list'.format(IOTest.io_test_dir))

        itp = self.list_columns[0].to_ndarray()
        itp.sort()
        ritp = retrieved_columns['int_tens_pdarray'].to_ndarray()
        ritp.sort()
        ihp = self.list_columns[1].to_ndarray()
        ihp.sort()
        rihp = retrieved_columns['int_hundreds_pdarray'].to_ndarray()
        rihp.sort()
        fp = self.list_columns[2].to_ndarray()
        fp.sort()
        rfp = retrieved_columns['float_pdarray'].to_ndarray()
        rfp.sort()

        self.assertEqual(4, len(retrieved_columns))
        self.assertTrue((itp == ritp).all())
        self.assertTrue((ihp == rihp).all())
        self.assertTrue((fp == rfp).all())      
        self.assertEqual(len(self.list_columns[3]), 
                         len(retrieved_columns['bool_pdarray']))    
        self.assertEqual(4, 
                      len(ak.get_datasets('{}/iotest_list_LOCALE0000'.format(IOTest.io_test_dir))))
    
    def testLsHdf(self):
        '''
        Creates 1..n files depending upon the number of arkouda_server locales, invokes the 
        ls_hdf method on an explicit file name reads the files and confirms the expected 
        message was returned.

        :return: None
        :raise: AssertionError if the h5ls output does not match expected value
        '''
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column'.format(IOTest.io_test_dir))
        message = ak.ls_hdf('{}/iotest_single_column_LOCALE0000'.format(IOTest.io_test_dir))
        self.assertIn('int_tens_pdarray         Dataset', message)

    def testReadHdf(self):
        '''
        Creates 2..n files depending upon the number of arkouda_server locales with two
        files each containing different-named datasets with the same pdarrays, reads the files
        with an explicit list of file names to the read_hdf method, and confirms the dataset 
        was returned correctly.

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        '''
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column'.format(IOTest.io_test_dir))
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column_dupe'.format(IOTest.io_test_dir))
        
        dataset = ak.read_hdf(dsetName='int_tens_pdarray', 
                    filenames=['{}/iotest_single_column_LOCALE0000'.format(IOTest.io_test_dir),
                               '{}/iotest_single_column_dupe_LOCALE0000'.format(IOTest.io_test_dir)])
        self.assertIsNotNone(dataset)

        with self.assertRaises(RuntimeError) as cm:
            ak.read_hdf(dsetName='in_tens_pdarray', 
                    filenames=['{}/iotest_single_column_LOCALE0000'.format(IOTest.io_test_dir),
                               '{}/iotest_single_column_dupe_LOCALE0000'.format(IOTest.io_test_dir)])       
        self.assertTrue('Dataset in_tens_pdarray not found in file' in  
                         cm.exception.args[0])
        
        with self.assertRaises(RuntimeError) as cm:
            ak.read_hdf(dsetName='int_tens_pdarray', 
                    filenames=['{}/iotest_single_colum_LOCALE0000'.format(IOTest.io_test_dir),
                               '{}/iotest_single_colum_dupe_LOCALE0000'.format(IOTest.io_test_dir)])       
        self.assertTrue('iotest_single_colum_LOCALE0000 not found' in  cm.exception.args[0])

    def testReadHdfWithGlob(self):
        '''
        Creates 2..n files depending upon the number of arkouda_server locales with two
        files each containing different-named datasets with the same pdarrays, reads the files
        with the glob feature of the read_hdf method, and confirms the datasets and embedded 
        pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        '''
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column'.format(IOTest.io_test_dir))
        self._create_file(columns=self.dict_single_column, 
                          prefix_path='{}/iotest_single_column_dupe'.format(IOTest.io_test_dir))
        
        dataset = ak.read_hdf(dsetName='int_tens_pdarray', 
                    filenames='{}/iotest_single_column*'.format(IOTest.io_test_dir))
        self.assertEqual(self.int_tens_pdarray.all(), dataset.all())

    def testReadAll(self):
        '''
        Creates 2..n files depending upon the number of arkouda_server locales, reads the files
        with an explicit list of file names to the read_all method, and confirms the datasets 
        and embedded pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns'.format(IOTest.io_test_dir))
        
        dataset = ak.read_all(filenames=['{}/iotest_dict_columns_LOCALE0000'.format(IOTest.io_test_dir)])
        self.assertEqual(4, len(list(dataset.keys())))     
        
    def testReadAllWithGlob(self):
        '''
        Creates 2..n files depending upon the number of arkouda_server locales with two
        files each containing different-named datasets with the same pdarrays, reads the files
        with the glob feature of the read_all method, and confirms the datasets and embedded 
        pdarrays match the input dataset and pdarrays

        :return: None
        :raise: AssertionError if the input and returned datasets don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns'.format(IOTest.io_test_dir))
         
        retrieved_columns = ak.read_all(filenames='{}/iotest_dict_columns*'.format(IOTest.io_test_dir))

        itp = self.list_columns[0].to_ndarray()
        itp.sort()
        ritp = retrieved_columns['int_tens_pdarray'].to_ndarray()
        ritp.sort()
        ihp = self.list_columns[1].to_ndarray()
        ihp.sort()
        rihp = retrieved_columns['int_hundreds_pdarray'].to_ndarray()
        rihp.sort()
        fp = self.list_columns[2].to_ndarray()
        fp.sort()
        rfp = retrieved_columns['float_pdarray'].to_ndarray()
        rfp.sort()

        self.assertEqual(4, len(list(retrieved_columns.keys())))  
        self.assertTrue((itp == ritp).all())
        self.assertTrue((ihp == rihp).all())
        self.assertTrue((fp == rfp).all())
        self.assertEqual(len(self.bool_pdarray), len(retrieved_columns['bool_pdarray']))

    def testLoad(self):
        '''
        Creates 1..n files depending upon the number of arkouda_server locales with three columns 
        AKA datasets, loads each corresponding dataset and confirms each corresponding pdarray 
        equals the input pdarray.
        
        :return: None
        :raise: AssertionError if the input and returned datasets (pdarrays) don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns'.format(IOTest.io_test_dir)) 
        result_array_tens = ak.load(path_prefix='{}/iotest_dict_columns'.format(IOTest.io_test_dir), 
                                    dataset='int_tens_pdarray')
        result_array_hundreds = ak.load(path_prefix='{}/iotest_dict_columns'.format(IOTest.io_test_dir), 
                                        dataset='int_hundreds_pdarray')
        result_array_floats = ak.load(path_prefix='{}/iotest_dict_columns'.format(IOTest.io_test_dir), 
                                     dataset='float_pdarray')
        result_array_bools = ak.load(path_prefix='{}/iotest_dict_columns'.format(IOTest.io_test_dir), 
                                     dataset='bool_pdarray')
        
        ratens = result_array_tens.to_ndarray()
        ratens.sort()
        
        rahundreds = result_array_hundreds.to_ndarray()
        rahundreds.sort()

        rafloats = result_array_floats.to_ndarray()
        rafloats.sort()

        self.assertTrue((self.int_tens_ndarray == ratens).all())
        self.assertTrue((self.int_hundreds_ndarray  == rahundreds).all())
        self.assertTrue((self.float_ndarray == rafloats).all())
        self.assertEqual(len(self.bool_pdarray), len(result_array_bools))
        
    def testGetDataSets(self):
        '''
        Creates 1..n files depending upon the number of arkouda_server locales containing three 
        datasets and confirms the expected number of datasets along with the dataset names
        
        :return: None
        :raise: AssertionError if the input and returned dataset names don't match
        '''
        self._create_file(columns=self.dict_columns, 
                          prefix_path='{}/iotest_dict_columns'.format(IOTest.io_test_dir))     
        datasets = ak.get_datasets('{}/iotest_dict_columns_LOCALE0000'.format(IOTest.io_test_dir))

        self.assertEqual(4, len(datasets)) 
        for dataset in datasets:
            self.assertIn(dataset, self.names)

    def testSaveStringsDataset(self):
        # Create, save, and load Strings dataset
        strings_array = ak.array(['testing string{}'.format(num) for num in list(range(0,25))])
        strings_array.save('{}/strings-test'.format(IOTest.io_test_dir), dataset='strings')
        r_strings_array = ak.load('{}/strings-test'.format(IOTest.io_test_dir), 
                                  dataset='strings')

        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = r_strings_array.to_ndarray()
        r_strings.sort()
        self.assertTrue((strings == r_strings).all())

        # Read a part of a saved Strings dataset from one hdf5 file
        r_strings_subset = ak.read_all(filenames='{}/strings-test_LOCALE0000'.\
                                    format(IOTest.io_test_dir))
        self.assertIsNotNone(r_strings_subset)
        self.assertTrue(isinstance(r_strings_subset[0], str))    
        self.assertIsNotNone(ak.read_hdf(filenames='{}/strings-test_LOCALE0000'.\
                            format(IOTest.io_test_dir), dsetName='strings/values'))
        self.assertIsNotNone(ak.read_hdf(filenames='{}/strings-test_LOCALE0000'.\
                            format(IOTest.io_test_dir), dsetName='strings/segments'))
     
    def testSaveLongStringsDataset(self):
        # Create, save, and load Strings dataset
        strings = ak.array(['testing a longer string{} to be written, loaded and appended'.\
                                  format(num) for num in list(range(0,26))])
        strings.save('{}/strings-test'.format(IOTest.io_test_dir), dataset='strings')

        n_strings = strings.to_ndarray()
        n_strings.sort()
        r_strings = ak.load('{}/strings-test'.format(IOTest.io_test_dir), 
                                  dataset='strings').to_ndarray()
        r_strings.sort()

        self.assertTrue((n_strings == r_strings).all())       

    def testSaveMixedStringsDataset(self):
        strings_array = ak.array(['string {}'.format(num) for num in list(range(0,25))])
        m_floats =  ak.array([x / 10.0 for x in range(0, 10)])      
        m_ints = ak.array(list(range(0, 10)))
        ak.save_all({'m_strings': strings_array,
                     'm_floats' : m_floats,
                     'm_ints' : m_ints}, 
                     '{}/multi-type-test'.format(IOTest.io_test_dir))
        r_mixed = ak.load_all('{}/multi-type-test'.format(IOTest.io_test_dir))

        self.assertTrue((strings_array.to_ndarray().sort() == \
                                           r_mixed['m_strings'].to_ndarray().sort()))
        self.assertIsNotNone(r_mixed['m_floats'])
        self.assertIsNotNone(r_mixed['m_ints'])

        r_floats = ak.sort(ak.load('{}/multi-type-test'.format(IOTest.io_test_dir), 
                           dataset='m_floats'))
        self.assertTrue((m_floats == r_floats).all())

        r_ints = ak.sort(ak.load('{}/multi-type-test'.format(IOTest.io_test_dir), 
                           dataset='m_ints'))
        self.assertTrue((m_ints == r_ints).all())
        
        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = ak.load('{}/multi-type-test'.format(IOTest.io_test_dir), 
                            dataset='m_strings').to_ndarray()
        r_strings.sort()

        self.assertTrue((strings == r_strings).all())

    def testAppendStringsDataset(self):
        strings_array = ak.array(['string {}'.format(num) for num in list(range(0,25))])
        strings_array.save('{}/append-strings-test'.format(IOTest.io_test_dir), 
                           dataset='strings')
        strings_array.save('{}/append-strings-test'.format(IOTest.io_test_dir), 
                           dataset='strings-dupe', mode='append')

        r_strings = ak.load('{}/append-strings-test'.format(IOTest.io_test_dir), 
                                 dataset='strings')
        r_strings_dupe = ak.load('{}/append-strings-test'.format(IOTest.io_test_dir), 
                                 dataset='strings-dupe')  
        self.assertTrue((r_strings == r_strings_dupe).all())

    def testAppendMixedStringsDataset(self):
        strings_array = ak.array(['string {}'.format(num) for num in list(range(0,25))])
        strings_array.save('{}/append-multi-type-test'.format(IOTest.io_test_dir), 
                           dataset='m_strings') 
        m_floats =  ak.array([x / 10.0 for x in range(0, 10)])      
        m_ints = ak.array(list(range(0, 10)))
        ak.save_all({'m_floats' : m_floats,
                     'm_ints' : m_ints}, 
                     '{}/append-multi-type-test'.format(IOTest.io_test_dir), mode='append')
        r_mixed = ak.load_all('{}/append-multi-type-test'.format(IOTest.io_test_dir))
        
        self.assertIsNotNone(r_mixed['m_floats'])
        self.assertIsNotNone(r_mixed['m_ints'])
 
        r_floats = ak.sort(ak.load('{}/append-multi-type-test'.format(IOTest.io_test_dir), 
                                   dataset='m_floats'))
        r_ints = ak.sort(ak.load('{}/append-multi-type-test'.format(IOTest.io_test_dir), 
                                 dataset='m_ints'))
        self.assertTrue((m_floats == r_floats).all())
        self.assertTrue((m_ints == r_ints).all())
        
        strings = strings_array.to_ndarray()
        strings.sort()
        r_strings = r_mixed['m_strings'].to_ndarray()
        r_strings.sort()

        self.assertTrue((strings  == r_strings).all())

    def testStrictTypes(self):
        N = 100
        prefix = '{}/strict-type-test'.format(IOTest.io_test_dir)
        inttypes = [np.uint32, np.int64, np.uint16, np.int16]
        floattypes = [np.float32, np.float64, np.float32, np.float64]
        for i, (it, ft) in enumerate(zip(inttypes, floattypes)):
            with h5py.File('{}-{}'.format(prefix, i), 'w') as f:
                idata = np.arange(i*N, (i+1)*N, dtype=it)
                f.create_dataset('integers', data=idata)
                fdata = np.arange(i*N, (i+1)*N, dtype=ft)
                f.create_dataset('floats', data=fdata)
        with self.assertRaises(RuntimeError) as cm:
            a = ak.read_all(prefix+'*')
        self.assertTrue('Inconsistent precision or sign' in cm.exception.args[0])
        a = ak.read_all(prefix+'*', strictTypes=False)
        self.assertTrue((a['integers'] == ak.arange(len(inttypes)*N)).all())
        self.assertTrue(np.allclose(a['floats'].to_ndarray(), np.arange(len(floattypes)*N, dtype=np.float64)))
    
    def testTo_ndarray(self):
        ones = ak.ones(10)
        n_ones = ones.to_ndarray()
        new_ones = ak.array(n_ones)
        self.assertTrue((ones.to_ndarray() == new_ones.to_ndarray()).all())
        
        empty_ones = ak.ones(0)
        n_empty_ones = empty_ones.to_ndarray()
        new_empty_ones = ak.array(n_empty_ones)
        self.assertTrue((empty_ones.to_ndarray() == new_empty_ones.to_ndarray()).all())

    def tearDown(self):
        super(IOTest, self).tearDown()
        for f in glob.glob('{}/*'.format(IOTest.io_test_dir)):
            os.remove(f)

    @classmethod
    def tearDownClass(cls):
        super(IOTest, cls).tearDownClass()
        shutil.rmtree(IOTest.io_test_dir)
