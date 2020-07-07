import numpy as np
class Point():
    """Opis punktu"""
    def __init__(self, nr='', x=0, y=0, xw=0, yw=0, vx=0, vy=0):
        self.nr = nr
        self.x = x
        self.y = y
        self.xw = xw
        self.yw = yw
        self.vx = vx
        self.vy = vy



#Wczytywanie punktów z pliku
def wczytaj(plik):
    tmp = []
    with open(plik, 'r+') as pl:
        linie = pl.readlines()
        for ln in linie:
            li = ln.rstrip().lstrip().split()
            pkt = Point(li[0], float(li[1]), float(li[2]))
            tmp.append(pkt)
    return tmp

#Wyszukiwanie punktu o danym numerze w zbiorze
def szukaj(nr,wsp):
    tmp = 'brak'
    for i in wsp:
        if str(i.nr) == str(nr):
            tmp = i
    return tmp

#Obliczenie parametrów transformacji wielomianowej II-go stopnia
def par_w2(P,W):
    Ax = []
    Ay = []
    Lx = []
    Ly = []
    for p in P:
        w = szukaj(p.nr,W)
        Ax.append([1, p.x, p.y, p.x * p.y, p.x ** 2, p.y ** 2, 0, 0, 0, 0, 0, 0])
        Ay.append([0, 0, 0, 0, 0, 0, 1, p.x, p.y, p.x * p.y, p.x ** 2, p.y ** 2])
        Lx.append(w.x)
        Ly.append(w.y)
    A = np.array(Ax + Ay)
    L = np.array(Lx + Ly)
    dX = np.linalg.inv(A.T @ A) @ A.T @ L
    v = np.array(A @ dX - L)
    X = A @ dX
    vx = v[0:len(Ax)]
    vy = v[len(Ax):]
    Xx = X[0:len(Ax)]
    Xy = X[len(Ax):]
    m0 = np.sqrt(((v.T @ v) / ( 2 * (len(Ax) - len(dX)) )))
    covX = (m0 ** 2) * np.linalg.inv(A.T @ A)
    covO = A @ covX @ A.T
    wtorne = []
    RMSx = np.sqrt( (vx @ vx) / len(vx) )
    RMSy = np.sqrt((vy @ vy) / len(vy))
    mx = np.sqrt( (vx @ vx) / (len(vx)-len(dX)/2) )
    my = np.sqrt((vy @ vy) / (len(vy)-len(dX)/2))
    svx = 0
    svy = 0
    for i in range(0, len(vx)):
        svx += (vx[i] - np.mean(vx)) ** 2
        svy += (vy[i] - np.mean(vy)) ** 2
    msrvx = np.sqrt(svx/(len(vx)-1))
    msrvy = np.sqrt(svy/(len(vy)-1))
    for i, p in enumerate(P):
        tmp = Point(p.nr, p.x, p.y, Xx[i], Xy[i], vx[i], vy[i])
        wtorne.append(tmp)
    return dX, wtorne,  covX, covO, [m0, mx, my, RMSx, RMSy, msrvx, msrvy, min(vx), max(vx), min(vy), max(vy)]

#Obliczenie współrzędnych punktów w układzie wtórnym na podstawie parametrów transformacji wielomianowej II-go stopnia
def wsp_w2(par, P):
    pkt = []
    for p in P:
        X = par[0] + par[1] * p.x + par[2] * p.y + par[3] *p.x * p.y + par[4] * (p.x ** 2) + par[5] * (p.y ** 2)
        Y = par[6] + par[7] * p.x + par[8] * p.y + par[9] * p.x * p.y + par[10] * (p.x ** 2) + par[11] * (p.y ** 2)
        tmp = Point(p.nr, X, Y)
        pkt.append(tmp)
    return pkt

#Obliczenie parametrów transformacji wielomianowej III-go stopnia
def par_w3(P,W):
    Ax = []
    Ay = []
    Lx = []
    Ly = []
    for p in P:
        w = szukaj(p.nr,W)
        Ax.append([1, p.x, p.y, p.x * p.y, p.x ** 2, p.y ** 2, (p.x ** 2) * p.y, p.x * (p.y ** 2), p.x ** 3,
                   p.y ** 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        Ay.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, p.x, p.y, p.x * p.y, p.x ** 2, p.y ** 2, (p.x ** 2) * p.y, p.x * (p.y ** 2), p.x ** 3,
                   p.y ** 3])
        Lx.append(w.x)
        Ly.append(w.y)
    A = np.array(Ax + Ay)
    L = np.array(Lx + Ly)
    dX = np.linalg.inv(A.T @ A) @ A.T @ L
    v = np.array(A @ dX - L)
    X = A @ dX
    vx = v[0:len(Ax)]
    vy = v[len(Ax):]
    Xx = X[0:len(Ax)]
    Xy = X[len(Ax):]
    m0 = np.sqrt(((v.T @ v) / ( 2 * (len(Ax) - len(dX)) )))
    covX = (m0 ** 2) * np.linalg.inv(A.T @ A)
    covO = A @ covX @ A.T
    wtorne = []
    RMSx = np.sqrt( (vx @ vx) / len(vx) )
    RMSy = np.sqrt((vy @ vy) / len(vy))
    mx = np.sqrt( (vx @ vx) / (len(vx)-len(dX)/2) )
    my = np.sqrt((vy @ vy) / (len(vy)-len(dX)/2))
    svx = 0
    svy = 0
    for i in range(0, len(vx)):
        svx += (vx[i] - np.mean(vx)) ** 2
        svy += (vy[i] - np.mean(vy)) ** 2
    msrvx = np.sqrt(svx/(len(vx)-1))
    msrvy = np.sqrt(svy/(len(vy)-1))
    for i, p in enumerate(P):
        tmp = Point(p.nr, p.x, p.y, Xx[i], Xy[i], vx[i], vy[i])
        wtorne.append(tmp)

    return dX, wtorne,  covX, covO, [m0, mx, my, RMSx, RMSy, msrvx, msrvy, min(vx), max(vx), min(vy), max(vy)]

#Obliczenie współrzędnych punktów w układzie wtórnym na podstawie parametrów transformacji wielomianowej III-go stopnia
def wsp_w3(par, P):
    pkt = []
    for p in P:
        X = par[0] + par[1] * p.x +par[2] * p.y +par[3] * p.x * p.y + par[4] * (p.x ** 2) +par[5] * (p.y ** 2)+par[6]\
            * (p.x ** 2) * p.y+par[7] * (p.y ** 2) * p.x+par[8] * (p.x ** 3)+par[9] * (p.y ** 3)

        Y = par[10] +par[11] * p.x +par[12] * p.y +par[13] * p.x * p.y +par[14] * (p.x ** 2) +par[15] * (p.y ** 2) +par[16] \
            * (p.x ** 2) * p.y +par[17] * (p.y ** 2) * p.x +par[18] * (p.x ** 3) +par[19] * (p.y ** 3)
        tmp = Point(p.nr, X, Y)
        pkt.append(tmp)
    return pkt

#Obliczenie parametrów transformacji afinicznej
def par_af(P,W):
    Ax = []
    Ay = []
    Lx = []
    Ly = []
    for p in P:
        w = szukaj(p.nr,W)
        Ax.append([1, p.x, p.y, 0, 0, 0])
        Ay.append([0, 0, 0, 1, p.x, p.y])
        Lx.append(w.x)
        Ly.append(w.y)
    A = np.array(Ax + Ay)
    L = np.array(Lx + Ly)
    dX = np.linalg.inv(A.T @ A) @ A.T @ L
    v = np.array(A @ dX - L)
    X = A @ dX
    vx = v[0:len(Ax)]
    vy = v[len(Ax):]
    Xx = X[0:len(Ax)]
    Xy = X[len(Ax):]
    m0 = np.sqrt(((v.T @ v) / ( 2 * (len(Ax) - len(dX)) )))
    covX = (m0 ** 2) * np.linalg.inv(A.T @ A)
    covO = A @ covX @ A.T
    wtorne = []
    RMSx = np.sqrt( (vx @ vx) / len(vx) )
    RMSy = np.sqrt((vy @ vy) / len(vy))
    mx = np.sqrt( (vx @ vx) / (len(vx)-len(dX)/2) )
    my = np.sqrt((vy @ vy) / (len(vy)-len(dX)/2))
    svx = 0
    svy = 0
    for i in range(0, len(vx)):
        svx += (vx[i] - np.mean(vx)) ** 2
        svy += (vy[i] - np.mean(vy)) ** 2
    msrvx = np.sqrt(svx/(len(vx)-1))
    msrvy = np.sqrt(svy/(len(vy)-1))
    for i, p in enumerate(P):
        tmp = Point(p.nr, p.x, p.y, Xx[i], Xy[i], vx[i], vy[i])
        wtorne.append(tmp)
    return dX, wtorne,  covX, covO, [m0, mx, my, RMSx, RMSy, msrvx, msrvy, min(vx), max(vx), min(vy), max(vy)]

#Obliczenie współrzędnych punktów w układzie wtórnym na podstawie parametrów transformacji afinicznej
def wsp_af(par, P):
    pkt = []
    for p in P:
        X = par[0] + par[1] * p.x + par[2] * p.y
        Y = par[3] + par[4] * p.x + par[5] * p.y
        tmp = Point(p.nr, X, Y)
        pkt.append(tmp)
    return pkt

#Zapis współrzędnych do pliku
def zapis_wsp(nazwa,p):
    pl = open(nazwa, 'w+')
    for i in p:
        pl.write('W2{}\t{:<.3f}\t{:<.3f}\n'.format(i.nr, i.xw, i.yw ))
    pl.close()

#Raport z transformacji afinicznej
def rap_af(sciezka, dok,jed_pier,jed_wto,W,pa,cov,bl, teraz):
    m = np.sqrt(np.diag(cov))
    pl = open(sciezka , 'w+')
    pl.write('\n\n\n\n')
    pl.write('<>' * 39)
    pl.write('\n')
    pl.write('{:^78}\n'.format('Loża Szyderców and Company'))
    pl.write('{:^78}\n'.format('PRZEDSTAWIA'))
    pl.write('{:^78}\n'.format('Transformacja afiniczna'))
    pl.write('Wykonali:\ninż. Damian Ozga\ninż. Kamil Olko\n')
    pl.write('{:^78}\n'.format('AGH 2020'))
    pl.write('<>' * 39)
    pl.write('\n')
    pl.write('*'*79)
    pl.write('\n{:^79s}\n'.format('PUNKTY DOSTOSOWANIE'))
    pl.write('*' * 79)
    pl.write('\n\n')
    pl.write('{:<7s}{:<13s}{:<13s}{:<13s}{:<13s}{:<10s}{:<9s}\n'.format('Nr','X_pier'+ jed_pier,'Y_pier'+jed_pier,
                                                                           'X_wt'+jed_wto,'Y_wt'+jed_wto,
                                                                           'Vx'+jed_wto,'Vy'+jed_wto))
    for i in W:
        if dok == 0:
            pl.write('{:<7s}{:<13.0f}{:<13.0f}{:<13.0f}{:<13.0f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
            pl.write('{:<10.1f}{:<9.1f}\n'.format(i.vx, i.vy))
        if dok == 1:
            pl.write('{:<7s}{:<13.1f}{:<13.1f}{:<13.1f}{:<13.1f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
            pl.write('{:<10.2f}{:<9.2f}\n'.format(i.vx, i.vy))
        if dok == 2:
            pl.write('{:<7s}{:<13.2f}{:<13.2f}{:<13.2f}{:<13.2f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
            pl.write('{:<10.3f}{:<9.3f}\n'.format(i.vx, i.vy))
        if dok == 3:
            pl.write('{:<7s}{:<13.3f}{:<13.3f}{:<13.3f}{:<13.3f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
            pl.write('{:<10.4f}{:<9.4f}\n'.format(i.vx, i.vy))
        if dok == 4:
            pl.write('{:<7s}{:<13.4f}{:<13.4f}{:<13.4f}{:<13.4f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
            pl.write('{:<10.5f}{:<9.5f}\n'.format(i.vx, i.vy))
    pl.write('\n')
    pl.write('*'*79)
    pl.write('\n{:^79s}\n'.format('PARAMETRY TRANSFORMACJI'))
    pl.write('*' * 79)
    pl.write('\n\n')
    pl.write('a = {:<20.7f}ma = {:<20.7f}\nb = {:<20.7f}mb = {:<20.7f}''\nc = {:<20.7f}mc = {:<20.7f}\nd = '
             '{:<20.7f}md = {:<20.7f}\ne = {:<20.7f}me = {:<20.7f}\nf = {:<20.7f}mf = {'':<20.7f}\n'.format(pa[0],
                                                   m[0],pa[1],m[1],pa[2],m[2],pa[3],m[3],pa[4],m[4],pa[5],m[5]))
    pl.write('\n')
    pl.write('*'*79)
    pl.write('\n{:^79s}\n'.format('OCENA DOKŁADNOŚCI'))
    pl.write('*' * 79)
    pl.write('\n\n')
    pl.write('Proces transformacji:\n')
    pl.write('m0 = {:<20.5f}\nmx = {:<20.5f}\nmy = {:<20.5f}\nRMS_x = {:<20.5f}\nRMS_y = {:<20.5f}\n'.format(bl[0],
                                                                                                             bl[1],bl[2],
                                                                                                  bl[3],bl[4]))
    pl.write('\nOdchylenia standardowe wartości średnich poprawek:\n')
    pl.write('Sigma srednie_vx = {:<20.5f}\nSigma srednie_vy = {:<20.5f}\n'.format(bl[5],bl[6]))
    pl.write('1.98*sigma sr_vx = {:<20.5f}\n1.98*sigma sr_vy = {:<20.5f}\n'.format(1.98*bl[5], 1.98*bl[6]))
    pl.write('1.98*sigma sr_vx = {:<20.5f}\n1.98*sigma sr_vy = {:<20.5f}\n\n'.format(1.98 * bl[5], 1.98 * bl[6]))
    pl.write('Wartości maksymalne i minimalne poprawek:\n')
    pl.write('Vx_min = {:<20.5f}\nVx_max = {:<20.5f}\nVy_min = {:<20.5f}\nVy_max = {:<20.5f}\n\n'.format(bl[7],
                                                                                                         bl[8],
                                                                                                         bl[9],
                                                                                                         bl[10], ))
    pl.write('Test Blanda - Altmana:\n\nOś X:\n')
    parx = 0
    if bl[7] >= -1.98 * bl[5]:
        pl.write('Vx_min >= -1.98*sigma_śr_Vx\n')
    else:
        pl.write('Vx_min < -1.98*sigma_śr_Vx\n')
        parx += 1
    if bl[8] <= 1.98 * bl[5]:
        pl.write('Vx_max >= 1.98*sigma_śr_Vx\n')
    else:
        pl.write('Vx_max < 1.98*sigma_śr_Vx\n')
        parx += 1
    if parx > 0:
        pl.write('Oś X nie spełnia testu')
    else:
        pl.write('Oś X spełnia test')

    pl.write('\n\nOś Y\n')
    pary = 0
    if bl[9] >= -1.98 * bl[6]:
        pl.write('Vx_min >= 1.98*sigma_śr_Vx\n')
    else:
        pl.write('Vx_min < 1.98*sigma_śr_Vx\n')
        pary += 1
    if bl[10] <= 1.98 * bl[6]:
        pl.write('Vx_max >= 1.98*sigma_śr_Vx\n')
    else:
        pl.write('Vx_max < 1.98*sigma_śr_Vx\n')
        pary += 1
    if pary > 0:
        pl.write('Oś Y nie spełnia testu\n')
    else:
        pl.write('Oś Y spełnia test\n')


    pl.write('\n')
    pl.write('<>' * 39)
    pl.write('\n{:22}Data wykonania {:02}.{:02}.{:04} {:02}:{:02}:{:02}\n'.format(' ', teraz.day,
                                                                           teraz.month, teraz.year, teraz.hour,
                                                                           teraz.minute, teraz.second))
    pl.write('<>' * 39)

#Raport z transformacji wielomianowej II-go stopnia:
def rap_w2(sciezka, dok,jed_pier,jed_wto,W,pa,cov,bl, teraz):
        m = np.sqrt(np.diag(cov))
        pl = open(sciezka , 'w+')
        pl.write('\n\n\n\n')
        pl.write('<>' * 39)
        pl.write('\n')
        pl.write('{:^78}\n'.format('Loża Szyderców and Company'))
        pl.write('{:^78}\n'.format('PRZEDSTAWIA'))
        pl.write('{:^78}\n'.format('Transformacja wielomianowa drugiego stopnia'))
        pl.write('Wykonali:\ninż. Damian Ozga\ninż. Kamil Olko\n')
        pl.write('{:^78}\n'.format('AGH 2020'))
        pl.write('<>' * 39)
        pl.write('\n')
        pl.write('*'*79)
        pl.write('\n{:^79s}\n'.format('PUNKTY DOSTOSOWANIE'))
        pl.write('*' * 79)
        pl.write('\n\n')
        pl.write('{:<7s}{:<13s}{:<13s}{:<13s}{:<13s}{:<10s}{:<9s}\n'.format('Nr','X_pier'+ jed_pier,'Y_pier'+jed_pier,
                                                                               'X_wt'+jed_wto,'Y_wt'+jed_wto,
                                                                               'Vx'+jed_wto,'Vy'+jed_wto))
        for i in W:
            if dok == 0:
                pl.write('{:<7s}{:<13.0f}{:<13.0f}{:<13.0f}{:<13.0f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.1f}{:<9.1f}\n'.format(i.vx, i.vy))
            if dok == 1:
                pl.write('{:<7s}{:<13.1f}{:<13.1f}{:<13.1f}{:<13.1f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.2f}{:<9.2f}\n'.format(i.vx, i.vy))
            if dok == 2:
                pl.write('{:<7s}{:<13.2f}{:<13.2f}{:<13.2f}{:<13.2f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.3f}{:<9.3f}\n'.format(i.vx, i.vy))
            if dok == 3:
                pl.write('{:<7s}{:<13.3f}{:<13.3f}{:<13.3f}{:<13.3f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.4f}{:<9.4f}\n'.format(i.vx, i.vy))
            if dok == 4:
                pl.write('{:<7s}{:<13.4f}{:<13.4f}{:<13.4f}{:<13.4f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.5f}{:<9.5f}\n'.format(i.vx, i.vy))
        pl.write('\n')
        pl.write('*'*79)
        pl.write('\n{:^79s}\n'.format('PARAMETRY TRANSFORMACJI'))
        pl.write('*' * 79)
        pl.write('\n\n')
        pl.write('a0 = {:<20.7f}m_a0 = {:<20.7f}\n'
                 'a1 = {:<20.7f}m_a1 = {:<20.7f}\n'
                 'a2 = {:<20.7f}m_a2 = {:<20.7f}\n'
                 'a3 = {:<20.7f}m_a3 = {:<20.7f}\n'
                 'a4 = {:<20.7f}m_a4 = {:<20.7f}\n'
                 'a5 = {:<20.7f}m_a5 = {:<20.7f}\n'
                 'b0 = {:<20.7f}m_b0 = {:<20.7f}\n'       
                 'b1 = {:<20.7f}m_b1 = {:<20.7f}\n'       
                 'b2 = {:<20.7f}m_b2 = {:<20.7f}\n'           
                 'b3 = {:<20.7f}m_b3 = {:<20.7f}\n'           
                 'b4 = {:<20.7f}m_b4 = {:<20.7f}\n'           
                 'b5 = {:<20.7f}m_b5 = {:<20.7f}\n'
                 .format(pa[0],m[0],pa[1],m[1],pa[2],m[2],pa[3],m[3],pa[4],m[4],pa[5],m[5],
                    pa[6],m[6],pa[7],m[7],pa[8],m[8],pa[9],m[9],pa[10],m[10],pa[11],m[11]))
        pl.write('\n')
        pl.write('*'*79)
        pl.write('\n{:^79s}\n'.format('OCENA DOKŁADNOŚCI'))
        pl.write('*' * 79)
        pl.write('\n\n')
        pl.write('Proces transformacji:\n')
        pl.write('m0={:<20.5f}\nmx={:<20.5f}\nmy={:<20.5f}\nRMS_x={:<20.5f}\nRMS_y={:<20.5f}\n'.format(bl[0],bl[1],bl[2],
                                                                                                      bl[3],bl[4]))
        pl.write('\nOdchylenia standardowe wartości średnich poprawek:\n')
        pl.write('Sigma srednie_vx={:<20.5f}\nSigma srednie_vy={:<20.5f}\n'.format(bl[5],bl[6]))
        pl.write('1.98*sigma sr_vx = {:<20.5f}\n1.98*sigma sr_vy = {:<20.5f}\n'.format(1.98 * bl[5], 1.98 * bl[6]))
        pl.write('1.98*sigma sr_vx = {:<20.5f}\n1.98*sigma sr_vy = {:<20.5f}\n\n'.format(1.98 * bl[5], 1.98 * bl[6]))
        pl.write('Wartości maksymalne i minimalne poprawek:\n')
        pl.write('Vx_min = {:<20.5f}\nVx_max = {:<20.5f}\nVy_min = {:<20.5f}\nVy_max = {:<20.5f}\n\n'.format(bl[7],
                                                                                                             bl[8],
                                                                                                             bl[9],
                                                                                                          bl[10],))
        pl.write('Test Blanda - Altmana:\n\nOś X:\n')
        parx=0
        if bl[7] >= -1.98 * bl[5]:
            pl.write('Vx_min >= -1.98*sigma_śr_Vx\n')
        else:
            pl.write('Vx_min < -1.98*sigma_śr_Vx\n')
            parx += 1
        if bl[8] <= 1.98 * bl[5]:
            pl.write('Vx_max >= 1.98*sigma_śr_Vx\n')
        else:
            pl.write('Vx_max < 1.98*sigma_śr_Vx\n')
            parx += 1
        if parx > 0:
            pl.write('Oś X nie spełnia testu')
        else:
            pl.write('Oś X spełnia test')

        pl.write('\n\nOś Y\n')
        pary=0
        if bl[9] >= -1.98 * bl[6]:
            pl.write('Vx_min >= 1.98*sigma_śr_Vx\n')
        else:
            pl.write('Vx_min < 1.98*sigma_śr_Vx\n')
            pary += 1
        if bl[10] <= 1.98 * bl[6]:
            pl.write('Vx_max >= 1.98*sigma_śr_Vx\n')
        else:
            pl.write('Vx_max < 1.98*sigma_śr_Vx\n')
            pary += 1
        if pary > 0:
            pl.write('Oś Y nie spełnia testu\n')
        else:
            pl.write('Oś Y spełnia test\n')

        pl.write('\n')
        pl.write('<>' * 39)
        pl.write('\n{:22}Data wykonania {:02}.{:02}.{:04} {:02}:{:02}:{:02}\n'.format(' ', teraz.day,
                                                                               teraz.month, teraz.year, teraz.hour,
                                                                               teraz.minute, teraz.second))
        pl.write('<>' * 39)

#Raport z transformacji wielomianowej III-go stopnia:
def rap_w3(sciezka, dok,jed_pier,jed_wto,W,pa,cov,bl, teraz):
        m = np.sqrt(np.diag(cov))
        pl = open(sciezka , 'w+')
        pl.write('\n\n\n\n')
        pl.write('<>' * 39)
        pl.write('\n')
        pl.write('{:^78}\n'.format('Loża Szyderców and Company'))
        pl.write('{:^78}\n'.format('PRZEDSTAWIA'))
        pl.write('{:^78}\n'.format('Transformacja wielomianowa trzeciego stopnia'))
        pl.write('Wykonali:\ninż. Damian Ozga\ninż. Kamil Olko\n')
        pl.write('{:^78}\n'.format('AGH 2020'))
        pl.write('<>' * 39)
        pl.write('\n')
        pl.write('*'*79)
        pl.write('\n{:^79s}\n'.format('PUNKTY DOSTOSOWANIE'))
        pl.write('*' * 79)
        pl.write('\n\n')
        pl.write('{:<7s}{:<13s}{:<13s}{:<13s}{:<13s}{:<10s}{:<9s}\n'.format('Nr','X_pier'+ jed_pier,'Y_pier'+jed_pier,
                                                                               'X_wt'+jed_wto,'Y_wt'+jed_wto,
                                                                               'Vx'+jed_wto,'Vy'+jed_wto))
        for i in W:
            if dok == 0:
                pl.write('{:<7s}{:<13.0f}{:<13.0f}{:<13.0f}{:<13.0f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.1f}{:<9.1f}\n'.format(i.vx, i.vy))
            if dok == 1:
                pl.write('{:<7s}{:<13.1f}{:<13.1f}{:<13.1f}{:<13.1f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.2f}{:<9.2f}\n'.format(i.vx, i.vy))
            if dok == 2:
                pl.write('{:<7s}{:<13.2f}{:<13.2f}{:<13.2f}{:<13.2f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.3f}{:<9.3f}\n'.format(i.vx, i.vy))
            if dok == 3:
                pl.write('{:<7s}{:<13.3f}{:<13.3f}{:<13.3f}{:<13.3f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.4f}{:<9.4f}\n'.format(i.vx, i.vy))
            if dok == 4:
                pl.write('{:<7s}{:<13.4f}{:<13.4f}{:<13.4f}{:<13.4f}'.format(i.nr, i.x, i.y, i.xw, i.yw))
                pl.write('{:<10.5f}{:<9.5f}\n'.format(i.vx, i.vy))
        pl.write('\n')
        pl.write('*'*79)
        pl.write('\n{:^79s}\n'.format('PARAMETRY TRANSFORMACJI'))
        pl.write('*' * 79)
        pl.write('\n\n')
        pl.write('a0 = {:<20.7f}m_a0 = {:<20.7f}\n'
                 'a1 = {:<20.7f}m_a1 = {:<20.7f}\n'
                 'a2 = {:<20.7f}m_a2 = {:<20.7f}\n'
                 'a3 = {:<20.7f}m_a3 = {:<20.7f}\n'
                 'a4 = {:<20.7f}m_a4 = {:<20.7f}\n'
                 'a5 = {:<20.7f}m_a5 = {:<20.7f}\n'
                 'a6 = {:<20.7f}m_a6 = {:<20.7f}\n'
                 'a7 = {:<20.7f}m_a7 = {:<20.7f}\n'
                 'a8 = {:<20.7f}m_a8 = {:<20.7f}\n'
                 'a9 = {:<20.7f}m_a9 = {:<20.7f}\n'
                 'b0 = {:<20.7f}m_b0 = {:<20.7f}\n'       
                 'b1 = {:<20.7f}m_b1 = {:<20.7f}\n'       
                 'b2 = {:<20.7f}m_b2 = {:<20.7f}\n'           
                 'b3 = {:<20.7f}m_b3 = {:<20.7f}\n'           
                 'b4 = {:<20.7f}m_b4 = {:<20.7f}\n'           
                 'b5 = {:<20.7f}m_b5 = {:<20.7f}\n'
                 'b6 = {:<20.7f}m_b2 = {:<20.7f}\n'
                 'b7 = {:<20.7f}m_b3 = {:<20.7f}\n'
                 'b8 = {:<20.7f}m_b4 = {:<20.7f}\n'
                 'b9 = {:<20.7f}m_b5 = {:<20.7f}\n'.format(pa[0],m[0],pa[1],m[1],pa[2],m[2],pa[3],m[3],pa[4],m[4],pa[5]
                                                           ,m[5],pa[6],m[6],pa[7],m[7],pa[8],
                         m[8],pa[9],m[9],pa[10],m[10],pa[11],m[11],pa[12], m[12],pa[13], m[13],pa[14], m[14],pa[15],
                         m[15],pa[16], m[16],pa[17], m[17],pa[18], m[18],pa[19], m[19]))
        pl.write('\n')
        pl.write('*'*79)
        pl.write('\n{:^79s}\n'.format('OCENA DOKŁADNOŚCI'))
        pl.write('*' * 79)
        pl.write('\n\n')
        pl.write('Proces transformacji:\n')
        pl.write('m0={:<20.5f}\nmx={:<20.5f}\nmy={:<20.5f}\nRMS_x={:<20.5f}\nRMS_y={:<20.5f}\n'.format(bl[0],bl[1],bl[2],
                                                                                                      bl[3],bl[4]))
        pl.write('\nOdchylenia standardowe wartości średnich poprawek:\n')
        pl.write('Sigma srednie_vx={:<20.5f}\nSigma srednie_vy={:<20.5f}\n'.format(bl[5],bl[6]))

        pl.write('1.98*sigma sr_vx = {:<20.5f}\n1.98*sigma sr_vy = {:<20.5f}\n\n'.format(1.98 * bl[5], 1.98 * bl[6]))
        pl.write('Wartości maksymalne i minimalne poprawek:\n')
        pl.write('Vx_min = {:<20.5f}\nVx_max = {:<20.5f}\nVy_min = {:<20.5f}\nVy_max = {:<20.5f}\n\n'.format(bl[7],
                                                                                                             bl[8],
                                                                                                             bl[9],
                                                                                                          bl[10],))
        pl.write('Test Blanda - Altmana:\n\nOś X:\n')
        parx=0
        if bl[7] >= -1.98 * bl[5]:
            pl.write('Vx_min >= -1.98*sigma_śr_Vx\n')
        else:
            pl.write('Vx_min < -1.98*sigma_śr_Vx\n')
            parx += 1
        if bl[8] <= 1.98 * bl[5]:
            pl.write('Vx_max >= 1.98*sigma_śr_Vx\n')
        else:
            pl.write('Vx_max < 1.98*sigma_śr_Vx\n')
            parx += 1
        if parx > 0:
            pl.write('Oś X nie spełnia testu')
        else:
            pl.write('Oś X spełnia test')

        pl.write('\n\nOś Y\n')
        pary=0
        if bl[9] >= -1.98 * bl[6]:
            pl.write('Vx_min >= 1.98*sigma_śr_Vx\n')
        else:
            pl.write('Vx_min < 1.98*sigma_śr_Vx\n')
            pary += 1
        if bl[10] <= 1.98 * bl[6]:
            pl.write('Vx_max >= 1.98*sigma_śr_Vx\n')
        else:
            pl.write('Vx_max < 1.98*sigma_śr_Vx\n')
            pary += 1
        if pary > 0:
            pl.write('Oś Y nie spełnia testu\n')
        else:
            pl.write('Oś Y spełnia test\n')


        pl.write('\n')
        pl.write('<>' * 39)
        pl.write('\n{:22}Data wykonania {:02}.{:02}.{:04} {:02}:{:02}:{:02}\n'.format(' ', teraz.day,
                                                                               teraz.month, teraz.year, teraz.hour,
                                                                               teraz.minute, teraz.second))
        pl.write('<>' * 39)

