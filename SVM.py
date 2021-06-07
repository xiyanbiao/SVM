# -*- coding: utf-8 -*-


import os,math
import numpy as np
from osgeo import gdal
import shapefile
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")

class R_W_IMAGE(object):
    def read_proj(self,inpath):
        #  定义影像元数据属性
        self.inpath = inpath
        self.ds = gdal.Open(inpath)
        self.col = self.ds.RasterXSize #列
        self.row = self.ds.RasterYSize #行
        self.band = self.ds.RasterCount
        self.geoTransform = self.ds.GetGeoTransform() #获取仿射矩阵信息,#0左上角X，1水平分辨率，2旋转（0上面为北）,3左上角y,4旋转（0上面为北）,5垂直分辨率
        self.proj = (self.ds).GetProjection()
        self.tiles_flag = 0  #
        
    def read_data(self,inpath):   #整景影像读取
        self.read_proj(inpath)
        self.image_data = (self.ds).ReadAsArray() #
        if self.image_data.ndim == 3:
            self.image_data = self.image_data.transpose((1,2,0))  
        
    def read_vector(self,inpath):       
        self.sf = shapefile.Reader(inpath)
        self.minX, self.minY, self.maxX, self.maxY = self.sf.bbox #矢量范围
        
    def array2raster(self, data,output):  #数组输出影像，路径名
        if os.path.exists(output):
            row_start = self.out_row_start
            col_start = self.out_col_start
            if self.out_band > 1:
                for i in range(self.out_band):
                    self.out_driver.GetRasterBand(i+1).WriteArray(data[i,:,:],col_start,row_start)
            else:
                self.out_driver.GetRasterBand(1).WriteArray(data,col_start,row_start)

#            self.out_driver.GetRasterBand(1).WriteArray(data,col_start,row_start)  #(data,列偏移，行偏移)
        else:
            self.out_data = data
            if self.out_data.ndim == 3:
                (self.out_band,self.out_row,self.out_col) = self.out_data.shape  #(行，列，波段)
            elif self.out_data.ndim == 2:
                self.out_band = 1
                (self.out_row,self.out_col) = self.out_data.shape  #
            else:
                print('message: the dimension is overflow!')
                
            if 'int8' in self.out_data.dtype.name:
                self.out_datatype = gdal.GDT_Byte
            elif 'int16' in self.out_data.dtype.name:
                self.out_datatype = gdal.GDT_UInt16
            elif 'float32' in self.out_data.dtype.name:
                self.out_datatype = gdal.GDT_Float32
            else:
                self.out_datatype = gdal.GDT_Float64 #判断数据类型  
            #写入文件    
            driver = gdal.GetDriverByName('GTiff')
            if self.tiles_flag == 0:
                self.out_driver = driver.Create(output,self.out_col,self.out_row,self.out_band,self.out_datatype)
            else:
                self.out_driver = driver.Create(output,self.col,self.row,self.out_band,self.out_datatype)
            self.out_driver.SetGeoTransform(self.geoTransform)
            self.out_driver.SetProjection(self.proj)
            if self.out_band > 1:
                for i in range(self.out_band):
                    self.out_driver.GetRasterBand(i+1).WriteArray(data[i,:,:],0,0)
            else:
                self.out_driver.GetRasterBand(1).WriteArray(data,0,0)
                
#%%
class CLIP_IMAGE_BY_VECTOR(R_W_IMAGE):
    #用矢量范围生成样本    
    def shp_sample(self,field_name):  #field_name是矢量中标注类别的字段名
        table_attr = self.sf.fields   #检索特殊字符导出属性值，制作样本标签
        for label_index in range(1,len(table_attr)):
            if field_name in table_attr[label_index]:   # Name 是矢量数据类别字段名，手动输入或者在矢量数据修改字段名
                break             
        if self.sf.shapeTypeName in list(['POINT','POINTZ','MULTIPOINT']):
            number_shape = len(self.sf.shapes())
            label = []
            samples = []
            for i in range(number_shape):
                point_cor_feature = np.array(self.sf.shape(i).points)
                minX = point_cor_feature[0,0]  #Y 纬度， X经度
                maxY = point_cor_feature[0,1]
                
                col_index, row_index = self.cal_row_col(self.geoTransform, minX, maxY)
                shapeRecords = self.sf.shapeRecords()
                attr = shapeRecords[i].record[label_index-1]  #读取属性时有问题，差一个单位索引
                try:
                    if isinstance(self.image_data):
                        if self.image_data.ndim == 3:
                            samples.append(self.image_data[:,row_index,col_index])
                        elif self.image_data.ndim == 2:
                            samples.append(self.image_data[row_index,col_index])
                except:
                    in_put = self.ds.ReadAsArray(xoff= col_index, yoff= row_index,xsize=1, ysize=1)
                    samples.append(in_put.reshape(in_put.shape[0]))       
                
                label.append(int(attr))
                
            samples = np.array(samples)
            
        self.samples = samples
        self.labels = label
          
    @staticmethod
    def cal_row_col(geoMatrix, x, y): #地理坐标的像素位置计算
        ulx = geoMatrix[0]
        uly = geoMatrix[3]
        xDist = geoMatrix[1]
        yDist = geoMatrix[5]
        pixel = int((x - ulx) / xDist)
        line = int((uly - y) / abs(yDist))
        return (pixel, line)

class CLASSIFICATION(CLIP_IMAGE_BY_VECTOR):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tiles_size = 1000   #分块大小
        print("test:", x, y)
        
    def class_svm(self, C = 1, kernel = 'rbf',gamma = 'auto'):
        clf = svm.SVC(C = C, kernel = kernel, gamma = gamma)
        clf.fit(self.samples, self.labels)
        
        tiles_size = self.tiles_size
        tiles_row = math.ceil(self.row/tiles_size)
        tiles_col = math.ceil(self.col/tiles_size)
        for i in range(tiles_row):   #行,分块
            for j in range(tiles_col):      #列
                if (i < (tiles_row-1)) and (j < (tiles_col-1)):
                    tiles_data = (self.ds).ReadAsArray(xoff=j*tiles_size, yoff=i*tiles_size,
                                          xsize=tiles_size, ysize=tiles_size)

                elif (i < (tiles_row-1)) and (j == (tiles_col-1)):
                    tiles_data = (self.ds).ReadAsArray(xoff=j*tiles_size, yoff=i*tiles_size, 
                                      xsize=self.col-j*tiles_size, ysize=tiles_size)
                    
                elif (i == (tiles_row-1)) and (j == (tiles_col-1)):
                    tiles_data = (self.ds).ReadAsArray(xoff=j*tiles_size, yoff=i*tiles_size, 
                                          xsize=self.col-j*tiles_size, ysize=self.row-i*tiles_size)
                    
                elif (i == (tiles_row-1)) and (j < (tiles_col-1)):
                    tiles_data = (self.ds).ReadAsArray(xoff=j*tiles_size, yoff=i*tiles_size, 
                                          xsize=tiles_size, ysize=self.row-i*tiles_size)
                
                (band,row,col) = tiles_data.shape
                tiles_data = tiles_data.transpose((1,2,0))
                tiles_data = tiles_data.reshape((row*col,band))   
                tiles_data = np.nan_to_num(tiles_data)
                result = clf.predict(tiles_data)
                result = result.reshape((row,col))
                self.out_row_start = i*tiles_size
                self.out_col_start = j*tiles_size
                self.array2raster(result,self.class_outfile)
                del tiles_data       #
    
    def Export_accuracy(self,pred_label,test_label, out_file): #精度验证，（预测标签，测试标签）
        con_matrix = confusion_matrix(test_label, pred_label)
        correct_number = np.diag(con_matrix).sum()
        overal_number = con_matrix.sum()
        
        col_sum = np.sum(con_matrix,axis = 0, keepdims = True)
        col_max = np.max(con_matrix,axis = 0, keepdims = True)
        Producer_accuracy = np.nan_to_num(np.round(col_max.astype(float)/col_sum,decimals = 2))
        con_matrix = np.concatenate((con_matrix,Producer_accuracy),axis = 0)
        
        row_sum = np.sum(con_matrix,axis = 1, keepdims = True)
        row_max = np.max(con_matrix,axis = 1, keepdims = True)
        User_accuracy = np.nan_to_num(np.round(row_max.astype(float)/row_sum,decimals = 2))
        con_matrix = np.concatenate((con_matrix,User_accuracy),axis = 1)
        
        overal_accuracy = round(correct_number.astype(float)/overal_number,3)
        Kappa = cohen_kappa_score(test_label, pred_label)
        
        if os.path.exists(out_file):
            os.remove(out_file)
        con_matrix[-1,-1] = overal_accuracy
        f_c = open(out_file,'w')
        f_c.write('                     精度评价报告' + '\n')
        f_c.write('混淆矩阵：' + '\n')
        for i in range(con_matrix.shape[0]):
            f_c.write('\n')
            for j in range(con_matrix.shape[1]):
                if i == con_matrix.shape[0]-1:
                    f_c.write('%-10s'%str(round(con_matrix[i,j],3)))
                else:
                    if j == con_matrix.shape[1]-1:
                        f_c.write('%-10s'%str(round(con_matrix[i,j],3)))
                    else:
                        f_c.write('%-10s'%str(int(con_matrix[i,j])))
        f_c.write('\n')
        f_c.write('\n')
        f_c.write('总体分类精度：' +  str('%.1f' %(overal_accuracy*100)) + '%' + '\n')
        f_c.write('Kappa系数：' +  str('%.3f' %(Kappa)) + '\n')
        f_c.close()
        
    @staticmethod   
    def gen_mask(data):
        image_data = data * 0 +1
        data = np.nan_to_num(image_data).astype(int)
        return data
    
    def work(self, vector_file, image_file, field_name, C, kernel, gamma, class_out):
        self.class_outfile = class_out
        self.read_proj(image_file)
        self.read_vector(vector_file)
        self.shp_sample(field_name)  # 生成样本
        self.tiles_flag = 1  #数组输出控制
        self.class_svm(C = C, kernel = kernel, gamma = gamma)
        self.out_driver = None
        
    def verification(self,image_file,vector_file,field_name, out_file):
        self.read_vector(vector_file)
        self.read_proj(image_file) 
        self.shp_sample(field_name)
        self.Export_accuracy(self.labels, self.samples,out_file)  # 精度验证文件存放在输出NDVI特征影像文件夹中，class_accuracy.txt
        self.out_driver = None
    
class NDVI_PROCESS(R_W_IMAGE):
    def __init__(self, x, y):
        self.x = x  
        self.y = y
        self.tiles_size = 1000
        print("test:", x, y)
        
    def NDVI_result(self):  #计算积分、二阶微分和振幅
        (band,height,width) = self.image_data.shape
        b1 = self.NDVI_ampl().reshape((1,height,width))
        b2 = self.NDVI_integ().reshape((1,height,width))
        b3 = self.NDVI_diff()
        out = np.concatenate((b1,b2,b3),axis = 0)
        return out

    def NDVI_ampl(self):  #振幅
        NDVI_max = self.image_data.max(axis = 0)
        NDVI_min = self.image_data.min(axis = 0)
        NDVI_amp = (NDVI_max - NDVI_min)/2
        return NDVI_amp

    def NDVI_integ(self): #积分
        out_ndvi_integ = self.image_data.sum(axis = 0)-(self.image_data[0,:] + self.image_data[-1,:])/2
        return out_ndvi_integ

    def NDVI_diff(self):  #二阶微分
        kernal = np.array([[0,1,0],[1,-4,1],[0,1,0]]).reshape((1,3,3))
        (band,height,width) = self.image_data.shape
        out_data = np.zeros((band,height,width), dtype = 'float')
        padding_data = np.pad(self.image_data,1,mode = 'edge')
        padding_data = padding_data[1:-1,:,:]

        for i in range(height):
            for j in range(width):
                window_data = padding_data[:,i:i+3,j:j+3]
                out_data[:,i,j] = (window_data * kernal).sum(axis = (1,2))
        return out_data
    def work(self, image_file,out_file):
        if os.path.exists(out_file):
            os.remove(out_file)
     
        self.read_proj(image_file)         #     
        self.tiles_flag = 1
        tiles_size = self.tiles_size
        tiles_row = math.ceil(self.row/tiles_size)
        tiles_col = math.ceil(self.col/tiles_size)
        for i in range(tiles_row):   #行,分块
#            geoTransform1[3] = geoTransform[3] + i * tiles_size*geoTransform[5]
            for j in range(tiles_col):      #列
#                tile_out_file = os.path.join(out_dir, (str(i) + str(j) + '.tif'))
                if (i < (tiles_row-1)) and (j < (tiles_col-1)):
                    self.image_data = (self.ds).ReadAsArray(xoff=j*tiles_size, yoff=i*tiles_size,
                                          xsize=tiles_size, ysize=tiles_size)
                elif (i < (tiles_row-1)) and (j == (tiles_col-1)):
                    self.image_data = (self.ds).ReadAsArray(xoff=j*tiles_size, yoff=i*tiles_size, 
                                      xsize=self.col-j*tiles_size, ysize=tiles_size)
                    
                elif (i == (tiles_row-1)) and (j == (tiles_col-1)):
                    self.image_data = (self.ds).ReadAsArray(xoff=j*tiles_size, yoff=i*tiles_size, 
                                          xsize=self.col-j*tiles_size, ysize=self.row-i*tiles_size)
                elif (i == (tiles_row-1)) and (j < (tiles_col-1)):
                    self.image_data = (self.ds).ReadAsArray(xoff=j*tiles_size, yoff=i*tiles_size, 
                                          xsize=tiles_size, ysize=self.row-i*tiles_size)
                self.out_row_start = i*tiles_size
                self.out_col_start = j*tiles_size
#                geoTransform1[0] = geoTransform[0] + j * tiles_size * geoTransform[1]
#                self.geoTransform = tuple(geoTransform1)
                self.array2raster(self.NDVI_result(), out_file)  # 
                
        self.out_driver = None
        
if __name__ == '__main__':   
  
    
    #%%
    
    #功能1、参数（植被指数）特征提取
    #输入1，     合成影像
    # image_file = r'F:\t.dat'
    
    # #主函数
    # NDVI_P = NDVI_PROCESS(1, 'feature')
    
    # #输出：特征影像
    # outFile = r'F:\feature.tif'
    # NDVI_P.work(image_file,outFile)
    # del NDVI_P
    
    #%%
    #功能2、SVM分类
    #输入1，特征影像
    image_file = r'D:\test.dat'
    #输入2，训练样本矢量数据（*.shp）
    vector_file = r'D:\test.shp'
    #输入3，矢量样本数据类别字段名
    field_name = 'test'
    #输入4,penalty parameter
    C = 100   #默认100
    #输入5,kernel type 
    kernel = 'rbf'  #默认‘rbf’
    #输入6,Gamma
    gamma = 'auto'  #默认‘auto’,可输入浮点型数字
    
    #主函数
    C_svm = CLASSIFICATION(1, 'test')  
    
    #输出：分类结果
    outFile = r'D:\class.tif'
    C_svm.work(vector_file,image_file,field_name, C, kernel, gamma, outFile)
    del C_svm
    
    #%%
    #功能3、精度验证 
    #输入1，分类结果影像
    image_file = r'D:\class.tif'
    #输入2，验证样本矢量数据（*.shp）
    test_vector_file = r'D:\voladation.shp'
    #输入3，矢量样本数据类别字段名
    field_name = 'class'
    
    #主函数
    T_svm = CLASSIFICATION(1, 'accuracy')
    
    #输出：精度验证文件
    outFile = r'D:\accuracy.txt'
    T_svm.verification(image_file,test_vector_file,field_name,outFile) 
    