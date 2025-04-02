from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

class Preprocess():
    def __init__(self):
        pass
        
    def read_file(self, filePath):
        # Read CSV file into a numpy object
        data = np.genfromtxt(filePath, dtype=None, delimiter=',')
        try:
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            x2 = data[:, 3]
            y2 = data[:, 4]
            z2 = data[:, 5]
        except Exception as e: 
            return str(e)
        return x,y,z,x2,y2,z2

    def interpolate(self, y):
        yy = y
        longitud=len(yy)
        x = np.arange(0, longitud)
        f = interp1d(x, yy)
        xnew = np.linspace(x[0], x[-1], 2530)        
        ynew = f(xnew)
        #plt.figure(figsize=(10,5))
        #plt.plot(x, y, 'o', xnew, ynew, '-')
        #plt.show()
        return ynew

    def maps_2(self, ruta, x, y, z, x2, y2, z2):
        Data = np.genfromtxt(ruta, delimiter=',')

        # Assign extracted column values to variables
        #xa, ya, za, xb, yb, zb = [Data[:-1, i] for i in [0,1,2,3,4,5]]    
        xa, ya, za, xb, yb, zb = x, y, z, x2, y2, z2
        
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

        params = {
        "Time (sec.)": tiempo_escalar,
        "Path Length (m.)": (PLD, PLI),
        "Depth Perception (m.)": (DPD, DPI),
        "Motion Smoothness (in m/s^3)": (MSD, MSI),
        "Average Speed (mm/s)": (Mean_SpeedD, Mean_SpeedI),
        "Average Acceleration (mm/s^2)": (Mean_AccD, Mean_AccI),
        "Idle Time (%)": (idleD, idleI),
        "Economy of Area (au.)": (A_PLD, A_PLI),
        "Economy of Volume (au.)": (A_VD, A_VI),
        "Bimanual Dexterity": BD,
        "Energy of Area (J/cm^2.)": (EndoEAD, EndoEAI),
        "Energy of Volume (J/cm^3.)": (EndoEVD, EndoEVI)
        }

        #return params
        return tiempo_escalar, PLD, DPD, MSD, Mean_AccD, Mean_AccD, idleD, A_PLD, A_VD, EndoEAD, EndoEVD, PLI, DPI, MSI, Mean_SpeedI, Mean_AccI, idleI, A_PLI, A_VI, EndoEAI, EndoEVI, BD
        #return tiempo_escalar, PLD, DPD, MSD, Mean_AccD, Mean_AccD, idleD, A_PLD, A_VD, EndoEAD, EndoEVD
    
    def maps_1(self, ruta, x, y, z, x2, y2, z2):
        Data = np.genfromtxt(ruta, delimiter=',')

        # Assign extracted column values to variables
        #xa, ya, za, xb, yb, zb = [Data[:-1, i] for i in [0,1,2,3,4,5]]    
        xa, ya, za, xb, yb, zb = x, y, z, x2, y2, z2
        
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

        params = {
        "Time (sec.)": tiempo_escalar,
        "Path Length (m.)": (PLD, PLI),
        "Depth Perception (m.)": (DPD, DPI),
        "Motion Smoothness (in m/s^3)": (MSD, MSI),
        "Average Speed (mm/s)": (Mean_SpeedD, Mean_SpeedI),
        "Average Acceleration (mm/s^2)": (Mean_AccD, Mean_AccI),
        "Idle Time (%)": (idleD, idleI),
        "Economy of Area (au.)": (A_PLD, A_PLI),
        "Economy of Volume (au.)": (A_VD, A_VI),
        "Bimanual Dexterity": BD,
        "Energy of Area (J/cm^2.)": (EndoEAD, EndoEAI),
        "Energy of Volume (J/cm^3.)": (EndoEVD, EndoEVI)
        }

        #return params
        return tiempo_escalar, PLD, DPD, MSD, Mean_AccD, Mean_AccD, idleD, A_PLD, A_VD, EndoEAD, EndoEVD

    def gen_graph(self, x, y, z, x2, y2, z2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x, y, z, c='blue')
        ax.plot(x2, y2, z2, c='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

'''
import json
obj = training_process()
filePath = "Datos_Transferencia/Dr. Fernando P.E_2023-05-22_14-27-57.csv"
data = obj.read_file(filePath)

try:
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    x2 = data[:, 3]
    y2 = data[:, 4]
    z2 = data[:, 5]

    maps_values = obj.maps(filePath, x, y, z, x2, y2, z2)
    # Convert the maps dictionary to a JSON string
    maps_values = json.dumps(maps_values)
    #xyz=np.array((maps_values), dtype='float32')
    #label=np.array((j), dtype='float32')
    print("\nFilename: ", filePath)
    print(maps_values)
except Exception as e: 
            print(str(e))

obj = training_process()
directory = obj.path

clases = ["Expertos", "Intermedios", "Novatos"]
array3d = []
arrayLabels = []
i = 0
j = 0

# LOADING DATA INTO NUMPY OBJECTS FOR MODEL BUILD
for clase in clases:
    path = os.listdir(os.path.join(directory, clase))
    for filename in path:
        filePath = directory + clase + "/" + filename
        data = obj.read_file(filePath)

        try:
            data[23:, 2] = 8 - data[23:, 2]
            data[23:, 5] = 8 - data[23:, 5]

            x = data[23:, 0]
            y = data[23:, 1]
            z = data[23:, 2]
            x2 = data[23:, 3]
            y2 = data[23:, 4]
            z2 = data[23:, 5]

            maps_values = obj.maps(filePath, x, y, z, x2, y2, z2)
            # Convert the maps dictionary to a JSON string
            maps_values = json.dumps(maps_values)
            #xyz=np.array((maps_values), dtype='float32')
            #label=np.array((j), dtype='float32')
            print("\nFilename: ",i, filePath)
            print(maps_values)

            #array3d.append(xyz)
            #arrayLabels.append(label)

        except Exception as e: 
            print(str(e))
        
        i += 1
    j += 1

# convert the list of arrays to a numpy array
array3d = np.array(array3d)
arrayLabels = np.array(arrayLabels).reshape((i,1))

print(array3d.shape)
print(arrayLabels.shape)

# print the shape of the resulting array
np.save('MapsData_Corte_Derecha.npy', array3d)
np.save('ClassData_Corte.npy', arrayLabels)
'''

