import numpy as np 
from scipy import signal

def maps(ruta):
    Data = np.genfromtxt(ruta, delimiter=',')

    # Assign extracted column values to variables
    xa, ya, za, xb, yb, zb = [Data[:-1, i] for i in [0,1,2,3,4,5]]    
    
    # Extracting last value of last column, which is Time
    tiempo_escalar = Data[:, -1][-1] if Data.size > 0 else None
    tiempo = Data[:, -1]

    # Converting cm to m
    xa, ya, za, xb, yb, zb = xa/100, ya/100, za/100, xb/100, yb/100, zb/100
        
    #EndoViS Path Length - Derecha, Izquierda
    PLD = np.sum(np.sqrt(np.diff(xa,1)**2 + np.diff(ya,1)**2 + np.diff(za,1)**2))
    PLI = np.sum(np.sqrt(np.diff(xb,1)**2 + np.diff(yb,1)**2 + np.diff(zb,1)**2))
    
    #EndoViS Depth Perception - Derecha, Izquierda
    DPD = np.sum(np.sqrt(np.diff(ya,1)**2 + np.diff(za,1)**2))
    DPI = np.sum(np.sqrt(np.diff(yb,1)**2 + np.diff(zb,1)**2))

    # Depth Perception Along Trocar - Derecha, Izquierda
    DPT1 = np.sum(np.abs(np.diff(np.sqrt((xa[:-1] + 5.5)**2 + (ya[:-1] + 9.85)**2 + (za[:-1] - 15)**2))))
    DPT2 = np.sum(np.abs(np.diff(np.sqrt((xb[:-1] + 15.5)**2 + (yb[:-1] + 9.85)**2 + (zb[:-1] - 15)**2))))

    # EndoViS Motion Smoothness
    h = np.mean(np.diff(tiempo))
    # Derecha
    jd = ((xa[3:] - 3 * xa[2:-1] + 3 * xa[1:-2] - xa[:-3]) / h ** 3) ** 2 + \
        ((ya[3:] - 3 * ya[2:-1] + 3 * ya[1:-2] - ya[:-3]) / h ** 3) ** 2 + \
        ((za[3:] - 3 * za[2:-1] + 3 * za[1:-2] - za[:-3]) / h ** 3) ** 2

    JD = np.sqrt(0.5 * np.sum(jd))
    cte = (tiempo_escalar ** 5) / (2 * PLD ** 2)
    MS_prevd = np.sum((np.diff(xa, 3) ** 2) + (np.diff(ya, 3) ** 2) + (np.diff(za, 3) ** 2))
    MSD = np.sqrt(cte * MS_prevd)
    # Izquierda
    Xv,Yv,Zv=xb,yb,zb
    ji = ((xb[3:] - 3 * xb[2:-1] + 3 * xb[1:-2] - xb[:-3]) / h ** 3) ** 2 + \
        ((yb[3:] - 3 * yb[2:-1] + 3 * yb[1:-2] - yb[:-3]) / h ** 3) ** 2 + \
        ((zb[3:] - 3 * zb[2:-1] + 3 * zb[1:-2] - zb[:-3]) / h ** 3) ** 2

    JI = np.sqrt(0.5 * np.sum(ji))
    cte = (tiempo_escalar ** 5) / (2 * PLI ** 2)
    MS_previ = np.sum(np.diff(Xv, 3) ** 2 + np.diff(Yv, 3) ** 2 + np.diff(Zv, 3) ** 2)
    MSI = np.sqrt(cte * MS_previ)

    # Resampleo de la señal a cada segundo
    num = round(len(xa)/30)
    f = round(len(xa)/num)
    variables = [xa, ya, za, xb, yb, zb]
    windows = [3.2, 2.6, 0.5, 1.5, 0.2, 0.0]
    resampled = [signal.resample_poly(var, 1, f, window=('kaiser', w)) for var, w in zip(variables, windows)]
    xxa, yya, zza, xxb, yyb, zzb = resampled
    #Se convierten los datos en centimetros 
    xxa, yya, zza = xxa*1000, yya*1000, zza*1000
    xxb, yyb, zzb = xxb*1000, yyb*1000, zzb*1000

    #EndoViS Average Speed (mm/s) - Derecha, Izquierda
    SpeedD = np.sqrt(np.diff(xxa,1)**2 + np.diff(yya,1)**2 + np.diff(zza,1)**2)
    Mean_SpeedD = np.mean(SpeedD)
    SpeedI = np.sqrt(np.diff(xxb,1)**2 + np.diff(yyb,1)**2 + np.diff(zzb,1)**2)
    Mean_SpeedI = np.mean(SpeedI)

    #EndoViS Average Acceleration (mm/s^2) - Derecha, Izquierda
    Accd = np.sqrt(np.diff(xxa,2)**2 + np.diff(yya,2)**2 + np.diff(zza,2)**2)
    Mean_AccD = np.mean(Accd)
    Acci = np.sqrt(np.diff(xxb,2)**2 + np.diff(yyb,2)**2 + np.diff(zzb,2)**2)
    Mean_AccI = np.mean(Acci)

    #EndoViS Idle Time (%) - Derecha, Izquierda
    idle1D = np.argwhere(SpeedD<=5)
    idleD =(len(idle1D)/len(SpeedD))*100
    idle1I = np.argwhere(SpeedI<=5)
    idleI =(len(idle1I)/len(SpeedI))*100

    #EndoViS Max. Area (m^2) - Derecha, Izquierda
    max_horD = max(xa)-min(xa)
    max_vertD = max(ya)-min(ya)
    MaxAreaD = max_vertD*max_horD
    max_horI = max(xb)-min(xb)
    max_vertI = max(yb)-min(yb)
    MaxAreaI = max_vertI*max_horI

    #EndoViS Max. Volume (m^3) - Derecha, Izquierda
    max_altD = max(za)-min(za)
    MaxVolD = MaxAreaD*max_altD
    max_altI = max(zb)-min(zb)
    MaxVolI = MaxAreaI*max_altI

    #EndoViS Area/PL : EOA - Derecha, Izquierda
    A_PLD = np.sqrt(MaxAreaD)/PLD
    A_PLI = np.sqrt(MaxAreaI)/PLI

    #EndoViS Volume/PL: EOV - Derecha, Izquierda
    A_VD =  MaxVolD**(1/3)/PLD
    A_VI =  MaxVolI**(1/3)/PLI
    
    #EndoViS Bimanual Dexterity
    b= np.sum((SpeedI - Mean_SpeedI)*(SpeedD - Mean_SpeedD))
    d= np.sum(np.sqrt(((SpeedI - Mean_SpeedI)**2)*((SpeedD - Mean_SpeedD)**2)));   
    BD = b/d

    #EndoViS Energia - Derecha, Izquierda
    EXa = np.sum(xa**2)
    EYa = np.sum(ya**2)
    EZa = np.sum(za**2)
    EndoEAD = (EXa+EYa)/(MaxAreaD*100) #J/cm^2
    EndoEVD = (EXa+EYa+EZa)/(MaxVolD*100) #J/cm^3

    EXb = np.sum(xb**2)
    EYb = np.sum(yb**2)
    EZb = np.sum(zb**2)
    EndoEAI = (EXb+EYb)/(MaxAreaI*100) #J/cm^2
    EndoEVI = (EXb+EYb+EZb)/(MaxVolI*100) #J/cm^3

    # Print parameters
    print("\nTiempo (s): ", tiempo_escalar)
    print("Path Length (m.): ", PLD,PLI)
    print("Depth Perception (m.): ", DPD,DPI)
    print("Depth Perception along trocar", DPT1,DPT2)
    print('Motion Smoothness 1: ', JD,JI)
    print('Motion Smoothness 2: ', MSD,MSI)
    print("Average Speed (mm/s): ", Mean_SpeedD, Mean_SpeedI)
    print("Average Acceleration (mm/s^2): ", Mean_AccD,Mean_AccI)
    print("Idle Time (%): ", idleD,idleI )
    print("Economy of Area (au.): ", A_PLD,A_PLI)
    print("Economy of Volume (au.): ", A_VD,A_VI)
    print("Bimanual Dexterity", BD)
    print("Energy of Area (J/cm^2.): ", EndoEAD, EndoEAI)
    print("Energy of Volume (J/cm^3.): ", EndoEVD, EndoEVI)

ruta="Datos_Transferencia/EmilyInterna.txt"
maps(ruta)