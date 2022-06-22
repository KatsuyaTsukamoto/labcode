# -*- coding: utf-8 -*-
'''---------モジュールのインポート--------'''
import cv2
import os
import numpy as np
import copy
import shutil
import datetime
import matplotlib.pylab as plt

'''-------------変更箇所---------------'''
readfile='TESTC' #読み込む画像フォルダ
th=30 #閾値
fps=10 #動画のフレームレート
condition=str(th)
after_image='/after_image' #処理後の画像フォルダ
video='/video' #動画フォルダ
width = 256 #画像の横のピクセル数
hight = 128 #画像の縦のピクセル数
depth = 155 #画像の奥行きの長さ
'''------------関数定義--------------------'''
'''<<<<<<<<<メイン関数>>>>>>>>>>>>'''
def main(date):
    '''___________フォルダ作成_____________'''
    makefolder(date)
    makefolder(date+'/'+readfile+condition+after_image)
    makefolder(date+'/'+readfile+condition+after_image+'/image_th')
    makefolder(date+'/'+readfile+condition+after_image+'/image_kukei')
    makefolder(date+'/'+readfile+condition+after_image+'/image_dst')
    makefolder(date+'/'+readfile+condition+video)

    '''_____________画像複製______________'''
    rename_number=rename(readfile,date+'/'+readfile+condition+'/rename')

    '''________画像処理＆液滴径算出__________'''
    img_oris, img_dsts, img_grays, img_kukeis,sizes = gazousyori(date,'after_image',rename_number,th)

    '''________液滴径をヒストグラム化__________'''
    make_hist(sizes)

    '''______元，2値化，マーキング画像を動画化___'''
    mp4convert(img_grays,fps,date+'/'+readfile+condition+'/'+video,'syori',code='mp4v',gray='y')
    mp4convert(img_dsts,fps,date+'/'+readfile+condition+'/'+video,'dst',code='mp4v',gray='y')
    mp4convert(img_oris,fps,date+'/'+readfile+condition+'/'+video,'oriz',code='mp4v',gray='n')
    mp4convert(img_kukeis,fps,date+'/'+readfile+condition+'/'+video,'kukei',code='mp4v',gray='n')


'''<<<<<ヒストグラム作成関数>>>>>>'''
def make_hist(sizes):
    fig= plt.figure(figsize=(8,6))
    plt.hist(sizes, bins=35, range=(0,30), rwidth=0.5)
    plt.rcParams['font.family']='Arial'
    plt.xlabel('Diameter [μm]')
    plt.ylabel('Count [-]')
    fig.savefig(date+'/'+readfile+condition+"/diameterhist.png")
    plt.close()


'''<<<<<日時関数>>>>>>>>'''
def date(readfile):
    now = datetime.datetime.now()
    time=now.strftime('%y%m%d')
    return time

'''<<<<<フォルダ作成関数>>>>>>>>'''
def makefolder(folder):
    os.makedirs(folder, exist_ok=True)#


'''<<<<<動画作成関数>>>>>>>>'''
def mp4convert(imgs,fps,out_path,out_name,code='H264',gray='y'):
    ori_path = os.getcwd()
    os.chdir(out_path)
    size = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'%s'%code)
    if gray== 'y' :
        video = cv2.VideoWriter('%s.mp4'%(out_name), fourcc, fps, (size[1],size[0]), False)
    else:
        video = cv2.VideoWriter('%s.mp4'%(out_name), fourcc, fps, (size[1],size[0]))

    for i in range(0,len(imgs)):
        video.write(imgs[i])
    video.release()
    os.chdir(ori_path)
    print('書き込み完了')


'''<<<<<画像複製関数>>>>>>>>'''
def rename(before,after):
    c=[]
    shutil.copytree(before,after)
    count=0
    dlis=sorted(os.listdir(before))
    dlised=np.array(dlis)
    if np.any(dlised=='.DS_Store')==True:
        dlis.remove('.DS_Store')

    for data_file in dlis:
        count+=1
        new_name = "rename_%04d.bmp"%(count)
        os.rename(after+'/'+data_file,after+'/'+new_name)
        c.append(count)
    maisuu = len(c)
    print('名前変更完了')
    return maisuu


'''<<<<<画像処理＆液滴径算出関数>>>>>>>>'''
def gazousyori(date,after_image,rename_number,th):
    img_grays, img_dsts, img_oris, img_kukeis, sizes,all_void,av_sizess = [],[],[],[],[],[],[]
    x, y, w, h, size = [],[],[],[],[]
    for i in range(1, rename_number+1,1):
        '''___________画像読み込み_____________'''
        img_ori = cv2.imread(date+'/'+readfile +condition + '/rename/rename_%04d.bmp' % i)
        img_oris.append(img_ori)
        img_syori = copy.copy(img_ori)

        '''___________画像をグレースケール化_____________'''
        img_gray = cv2.cvtColor(img_syori, cv2.COLOR_BGR2GRAY)
        dst = cv2.fastNlMeansDenoising(img_gray,10,7,21)
        img_dsts.append(dst)
        cv2.imwrite(date+'/'+readfile+condition+'/'+after_image+'/image_dst'+'/image_dst-%04d.bmp' % i, dst)

        '''___________画像を2値化_____________'''
        ret,thresh1 = cv2.threshold(img_gray,th,255,cv2.THRESH_BINARY)
        img_grays.append(thresh1)
        cv2.imwrite(date+'/'+readfile+condition+'/'+after_image+'/image_th/image_th-%04d.bmp' % i, thresh1)

        '''___________ラベリング＆オブジェクトサイズ算出_____________'''
        nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(thresh1)


        '''___________液滴径計算＆除去_____________'''
        for j in range(0,len(data)):
            x, y, w, h, size = data[j,0],data[j,1],data[j,2],data[j,3],data[j,4]
            if size<=5 :
                continue
            if size>=500:
                continue
            size_pix=size*0.81
            dia_pix=np.sqrt(4*size_pix/np.pi)
            img_kukei = cv2.rectangle(img_syori,(x-1,y-1),(x+w,y+h),(0,255,0),1)
            sizes.append(dia_pix)
        if len(sizes)==0:
            continue

        '''___________液滴濃度計算_____________'''
        lvol=list(map(lambda x:4*np.pi*x**3/(3*8),sizes))
        lvol_sum=sum(lvol)
        vol=width*hight*depth*16
        void=lvol_sum/vol*100
        all_void.append(void)
        av_sizes=sum(sizes)/len(sizes)
        av_sizess.append(av_sizes)

        '''___________マーキング画像保存_____________'''
        img_kukeis.append(img_kukei)
        cv2.imwrite(date+'/'+readfile+condition+'/'+after_image+'/image_kukei/image_kukei-%04d.bmp' % i, img_kukei)

    '''___________平均液滴径＆濃度保存_____________'''
    f=open(date+'/'+readfile+condition+"/"+"a_condition.txt","w")
    f.write("平均液滴径[μm]:"+str(sum(av_sizess)/len(av_sizess))+"μm\n")
    f.write("平均濃度[%]:"+str(sum(all_void)/len(all_void))+"\n")
    f.close()
    print ('画像処理完了')
    return img_oris, img_dsts, img_grays, img_kukeis,sizes



'''-----------メイン関数実行----------------'''
if __name__ == '__main__':
    date=date(readfile)
    main(date)
