# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:27:11 2019

@author: Ye Anbang
"""

import xml.sax
import os

class MyObjects:
    def __init__(self):
        self.name = ""
        self.polygon = []
    
class XmlHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.file_name = ""
        self.width = 0
        self.height = 0
        self.object = []
   
     # 元素开始事件处
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "object":
            self.object.append(MyObjects())
        elif tag == "pt":
            self.object[-1].polygon.append([0,0])
 
    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "filename":
            self.file_name = content
        elif self.CurrentData == "nrows":
            self.height = int(content)
        elif self.CurrentData == "ncols":
            self.width = int(content)
        elif self.CurrentData == "name":
            self.object[-1].name = content
        elif self.CurrentData == "x":
            self.object[-1].polygon[-1][0] = int(content)
        elif self.CurrentData == "y":
            self.object[-1].polygon[-1][1] = int(content)
         
def parse_annotation_info(file_path):
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
 
    # 重写 ContextHandler
    Handler = XmlHandler()
    parser.setContentHandler( Handler )
    if os.path.exists(file_path):
        parser.parse(file_path)
        #print(Handler.file_name)
        #print(Handler.height)
        #print(Handler.width)
        #for obj in Handler.object:
            #print(obj.name, ": ", obj.polygon)
           
        return Handler
    else:
        raise Exception("file_path:%s Not Exist!"%file_path)
   

   
       