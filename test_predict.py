import paddle.fluid as fluid
from PIL import Image
import numpy as np
import cv2
import img_preprocess
import time

#创建执行器
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

save_path = 'models/vgg16_100ep/best_model'
# save_path = 'models/mobilenet/best_model'
# save_path = 'models/resnet50'
# save_path = 'models/resnet50_20ep/best_model'
#从模型中得到预测程序，输入数据名称列表和分类器
[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(
    dirname=save_path, executor=exe)

#定义一个图像预处理的函数


def load_image(file):
    # im = Image.open(file)
    im = file
    # im = im.resize((224, 224), Image.ANTIALIAS)
    im = cv2.resize(im,(224,224))
    im = np.array(im).astype(np.float32)
    #PIL打开图片存储顺序为H（高度），W（宽度），C（通道数）
    #但是PaddlePaddle要求数据的顺序是CHW，需要调整顺序
    im = im.transpose((2, 0, 1))
    #cifar训练的图片通道顺序为B(蓝)，G(绿)，R(红)
    #但是PIL打开的图片是默认的RGB，所以要交换通道
    im = im[(2, 1, 0), :, :]
    im = im/255.0  # 归一化
    im = np.expand_dims(im, axis=0)
    return im



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while 1:
    success, frame = cap.read()
    # cv2.imshow("orig", frame)
    # frame = img_preprocess.hisEqulColor(frame)
    # cv2.imshow("pre",frame)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #获取图片数据
    img = load_image(frame)
    if success:
        start = time.time()
        #执行预测
        result = exe.run(program=infer_program, feed={
                 feeded_var_names[0]: img}, fetch_list=target_var)
        end = time.time()
        #获得值最大的元素对应的下标
        fps = 1 / (end - start)
        cv2.putText(frame, "FPS:{}".format(fps), (500, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        lab = np.argmax(result[0][0])

        names = ['normal', 'phone', 'disct',"3","4","5"]

        # print(np.shape(result)) 
        # print("预测的标签是：%d,名称是：%s,概率是：%f" % (lab, names[lab], result[0][0][lab]))

        cv2.putText(frame, "{} {} {}".format(lab, names[lab], result[0][0][lab]), (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # if result[0][0][0]> 0.9:
        #     cv2.putText(frame, "{}".format("normal"), (10, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # else:
        #     if result[0][0][1] > 0.4:
        #         cv2.putText(frame, "{}".format("phone"), (10, 70),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        #     elif result[0][0][2] > 0.6:
        #         cv2.putText(frame, "{}".format("dis"), (10, 90),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        #     else:
        #         cv2.putText(frame, "{}".format("normal"), (10, 50),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("test", frame)
    elif success == -1:
        print("调用失败")
    else:
        print('no result')
    key = cv2.waitKey(5)
    if key == 32:
        break
cv2.destroyAllWindows()


