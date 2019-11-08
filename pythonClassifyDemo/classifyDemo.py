#coding=utf-8

import hiai
from hiai.nn_tensor_lib import DataType
import imageNetClasses
import jpegHandler
import os
import numpy as np
import time
import cv2 as cv
import pickle

omFileName='./models/faster_rcnn.om'
srcFileDir = './ImageNetRaw/'
dstFileDir = './Result/'
kCategoryIndex = 2

def CreateGraph(model,modelInWidth,modelInHeight,dvppInWidth,dvppInHeight):

	myGraph = hiai.hiai._global_default_graph_stack.get_default_graph()
	if myGraph is None :
		print 'get defaule graph failed'
		return None
	print 'dvppwidth %d, dvppheight %d'%(dvppInWidth,dvppInHeight)
		
	cropConfig = hiai.CropConfig(0,0,dvppInWidth,dvppInHeight)
	print 'cropConfig ', cropConfig 
	resizeConfig = hiai.ResizeConfig(modelInWidth,modelInHeight)
	print 'resizeConfig ', resizeConfig

	nntensorList=hiai.NNTensorList()
	print 'nntensorList', nntensorList	

	resultCrop = hiai.crop(nntensorList,cropConfig)
	print 'resultCrop', resultCrop

	resultResize = hiai.resize(resultCrop, resizeConfig)
	print 'resultResize', resultResize

	resultInference = hiai.inference(resultResize, model, None)
	print 'resultInference', resultInference

	if ( hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
		print 'create graph ok !!!!'
		return myGraph
	else :
		print 'create graph failed, please check Davinc log.'
		return None

def CreateGraphWithoutDVPP(model):

	print model
	myGraph = hiai.hiai._global_default_graph_stack.get_default_graph()
	print myGraph
	if myGraph is None :
		print 'get defaule graph failed'
		return None


	nntensorList=hiai.NNTensorList()
	print nntensorList

	resultInference = hiai.inference(nntensorList, model, None)
	print nntensorList
	print hiai.HiaiPythonStatust.HIAI_PYTHON_OK
	#print myGraph.create_graph()

	if ( hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
		print 'create graph ok !!!!'
		return myGraph
	else :
		print 'create graph failed, please check Davinc log.'
		return None


def GraphInference(graphHandle,inputTensorList):
	if not isinstance(graphHandle,hiai.Graph) :
		print "graphHandle is not Graph object"
		return None

	resultList = graphHandle.proc(inputTensorList)
	return resultList

def PostProcess(resultList, dstFilePath, fileName):
    if resultList is not None:
        tensor_num = np.reshape(resultList[0], 32)
        tensor_bbox = np.reshape(resultList[1], (64, 304, 8))
        img = cv.imread(fileName)
        img_rows, img_cols, img_channel = img.shape
        scale_width = img_cols / float(896)
        scale_height = img_rows / float(608)
        bboxes = []
        for attr in range(32):
            num = int(tensor_num[attr])
            for bbox_idx in range(num):
                class_idx = attr * kCategoryIndex
                lt_x = scale_width * tensor_bbox[class_idx][bbox_idx][0]
                lt_y = scale_height * tensor_bbox[class_idx][bbox_idx][1]
                rb_x = scale_width * tensor_bbox[class_idx][bbox_idx][2]
                rb_y = scale_height * tensor_bbox[class_idx][bbox_idx][3]
                score = tensor_bbox[class_idx][bbox_idx][4]
                bboxes.append([int(lt_x), int(lt_y), int(rb_x), int(rb_y), attr, score])

        print('bboxes',bboxes)
        if len(bboxes) == 0:
            return None
        for box in bboxes:
            cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 4)
            cv.putText(img, "class:%d,score:%f" % (box[4], box[5]), (box[0], box[1]),
                        cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        _, filename = os.path.split(fileName)
        cv.imwrite("%s%s"%(dstFilePath, filename),img)
        return None
    else:
        print 'graph inference failed '
        return None

def main():
	inferenceModel = hiai.AIModelDescription('faster-rcnn',omFileName)
	print omFileName
	print inferenceModel
	# we will resize the jpeg to 896*608 to meet faster-rcnn requirement via opencv,
	# so DVPP resizing is not needed  	
	myGraph = CreateGraphWithoutDVPP(inferenceModel)
	if myGraph is None :
		print "CreateGraph failed"
		return None

	# in this sample demo, the faster-rcnn  model requires 896*608 images
	dvppInWidth = 896
	dvppInHeight = 608
	
	
	start = time.time()
	index=0
	pathDir =  os.listdir(srcFileDir)
	for allDir in pathDir :
		child = os.path.join('%s%s' % (srcFileDir, allDir))
		if( not jpegHandler.is_img(child) ):
			print '[info] file : ' + child + ' is not image !'
			continue 

		# read the jpeg file and resize it to required w&h, than change it to YUV format.	
		input_image = jpegHandler.jpeg2yuv(child, dvppInWidth, dvppInHeight)
                
		inputImageTensor = hiai.NNTensor(input_image,dvppInWidth,dvppInHeight,3,'testImage',DataType.UINT8_T, dvppInWidth*dvppInHeight*3/2)
		imageinfo=np.array([896,608,3]).astype(np.float32)
		imageinfo=np.reshape(imageinfo,(1,3))
		infoTensor = hiai.NNTensor(imageinfo, 1,3,1,'testinfo',DataType.FLOAT32_T, imageinfo.size)

		datalist=[inputImageTensor, infoTensor]
		nntensorList=hiai.NNTensorList(datalist)
		if not nntensorList:
			print "nntensorList is null"
			break
		resultList = GraphInference(myGraph,nntensorList)
		
		if resultList is None :
			print "graph inference failed"
			continue
		print resultList[1].shape
		PostProcess(resultList, dstFileDir,child )
	end = time.time()
	print 'cost time ' + str((end-start)*1000) + 'ms'		
	

	hiai.hiai._global_default_graph_stack.get_default_graph().destroy()

	

	print '-------------------end'
	

if __name__ == "__main__":
	main()
