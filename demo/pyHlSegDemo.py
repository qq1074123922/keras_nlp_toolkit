#coding=utf-8
import jpype
import os


if __name__ == "__main__":
    #打开jvm虚拟机
    jar_path = os.path.abspath('lib')
    # jvmPath = jpype.getDefaultJVMPath()
    jvmPath = "/Library/Java/JavaVirtualMachines/jdk-9.0.4.jdk/Contents/Home/lib/server/libjvm.dylib"
    jpype.startJVM(jvmPath, "-ea", "-Djava.class.path=%s" % (jar_path + '/hlSegment-2.1.10.jar'))
    # jpype.startJVM(jvmPath, "-ea", "-Djava.class.path=%s" % (jar_path + '/hlSegment-2.1.10.jar'), "-Djava.ext.dirs=%s" % jar_path)
    system = jpype.JClass('java.lang.System')
    system.out.println("hello, world")
    # BasicSegmentor = jpype.JClass('com.hylanda.segmentor.BasicSegmentor')

    #取得类定义
    # BasicSegmentor = jpype.JClass('com.hylanda.segmentor.BasicSegmentor')
    # SegOption = jpype.JClass('com.hylanda.segmentor.common.SegOption')
    # SegResult = jpype.JClass('com.hylanda.segmentor.common.SegResult')
    #
    # #创建分词对象
    # segmentor = BasicSegmentor()
    #
    # #加载词典
    # if not segmentor.loadDictionary("./dictionary/CoreDict.dat", None):
    #     print ("字典加载失败！")
    #     exit()
    #
    # #创建SegOption对象，如果使用默认的分词选项，也可以直接传空
    # option = SegOption()
    # option.mergeNumeralAndQuantity = False
    #
    # #分词
    # segResult = segmentor.segment(u"欢迎使用海量中文智能分词", option)
    #
    # #遍历分词结果
    # word = segResult.getFirst()
    # result = ""
    # while word != None:
    #     result += word.wordStr + ' '
    #     word = word.next
    #
    #
    # #输出结果
    # print (result)
    # jpype.shutdownJVM()
    # exit()
