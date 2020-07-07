from fun import *
import datetime



if __name__ == '__main__':
    jed_pier = '[pix]'
    jed_wto = '[mm]'
    P = wczytaj('P.txt')
    W = wczytaj('W.txt')
    teraz = datetime.datetime.now()

    #Obliczenie parametrów transformacji
    paa, wtornea, covXa, covOa, ma = par_af(P, W)
    pa2, wtorne2, covX2, covO2, m2 = par_w2(P, W)
    pa3, wtorne3, covX3, covO3, m3 = par_w3(P, W)


    #Zapis współrzędnych do pliku
    zapis_wsp('wsp_af',wtornea)
    zapis_wsp('wsp_w2', wtorne2)
    zapis_wsp('wsp_w3', wtorne3)

    #Zapis raportu do pliku
    rap_af('rapaf.txt',2,jed_pier,jed_wto,wtornea,paa,covXa,ma,teraz)
    rap_w2('rapw2.txt',2,jed_pier,jed_wto,wtorne2,pa2,covX2,m2,teraz)
    rap_w3('rapw3.txt',2, jed_pier, jed_wto, wtorne3, pa3, covX3, m3, teraz)


