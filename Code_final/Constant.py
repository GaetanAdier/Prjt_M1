# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 16:56:55 2015
@author: etienne
"""
import numpy as np

class stdIlluminant:
    """This class contains the standard illuminant referenced in the :py:class:`Constant.MatPass`.    
    
    :ivar A: CIE standard illuminant A (domestic tungsten filament lighting)
    :ivar C: CIE standard illuminant C (daylight simulator: average day light) -DEPRECATED
    :ivar D50: CIE standard illuminant D50 (horizon light)
    :ivar D55: CIE standard illuminant D55 (Mid-morning/afternoon daylight)
    :ivar D65: CIE standard illuminant D65 (Noon daylight)
    :ivar D75: CIE standard illuminant D75 (North sky daylight)
    :ivar E: CIE standard illuminant E (equal energy radiator)
    :ivar F2: CIE standard illuminant F2 (Cool white fluorescent)
    :ivar F7: CIE standard illuminant F7 (D65 simulator)
    :ivar F11: CIE standard illuminant F11 (Philips TL84, Ultralume 40)
    """
    A =   [1.09850, 1.00000, 0.35585]
    C =   [0.98074, 1.00000, 1.18232]
    D50 = [0.96422, 1.00000, 0.92149]
    D55 = [0.95682, 1.00000, 0.35585]
    D65 = [0.95047, 1.00000, 1.08883]
    D75 = [0.94972, 1.00000, 1.22638]
    E =   [0.3291 , 0.33333, 0.33333]
    F2 =  [0.99187, 1.00000, 0.67395]
    F7 =  [0.95041, 1.00000, 1.08747]
    F11 = [1.00966, 1.00000, 0.64370]
    
    
    
class MatPass:
    """ 
    
    This class contains all the transition matrix from RGB to XYZ space 
    
    :ivar AdobRGB: Adobe RGB, use with D65 standard illuminant and :math:`\gamma` = 2.2
    :ivar AppleRGB: Apple RGB, use with D65 standard illuminant and :math:`\gamma` = 1.8
    :ivar BestRGB: Best RGB, use with D50 standard illuminant and :math:`\gamma` = 2.2
    :ivar BetaRGB: Beta RGB, use with D50 standard illuminant and :math:`\gamma` = 2.2
    :ivar BruceRGB: Bruce RGB, use with D65 standard illuminant and :math:`\gamma` = 2.2
    :ivar CIERGB: CIE RGB, use with E standard illuminant and :math:`\gamma` = 2.2
    :ivar CmatchRGB: ColorMatch RGB, use with D50 standard illuminant and :math:`\gamma` = 1.8
    :ivar DonRGB4: Don RGB4, use with D50 standard illuminant and :math:`\gamma` = 2.2
    :ivar ECIRGB: ECI RGB, use with D50 standard illuminant and :math:`\gamma` = L
    :ivar ESPS5: Ekta Space PS5, use with D50 standard illuminant and :math:`\gamma` = 2.2
    :ivar NTSCRGB: NTSC RGB, use with C standard illuminant and :math:`\gamma` = 2.2
    :ivar PSRGB: Pal/secam RGB, use with D65 standard illuminant and :math:`\gamma` = 2.2
    :ivar PPRGB: Prophoto RGB, use with D50 standard illuminant and :math:`\gamma` = 1.8
    :ivar SMPT_CRGB: SMPTE-C RGB, use with D65 standard illuminant and :math:`\gamma` = 2.2
    :ivar sRGB: sRGB, use with D65 standard illuminant and :math:`\gamma` = 2.2
    :ivar WGRGB: Wide Gamut RGB, use with D50 standard illuminant and :math:`\gamma` = 2.2
    
    """
    
    AdobRGB =  [[0.5767309 , 0.1855540 , 0.1881852],[0.2973769 , 0.6273491 , 0.0752741],[0.0270343 , 0.0706872 , 0.9911085]] #: Adobe RGB (D65)
    AdobRGB=np.array(AdobRGB) #: Adobe RGB (D65)
    #: Apple RGB (D65)
    AppleRGB = [[0.4497288 , 0.3162486 , 0.1844926],[0.2446525 , 0.6720283 , 0.0833192],[0.0251848 , 0.1411824 , 0.9224628]]
    AppleRGB=np.array(AppleRGB)
    #Best RGB (D50)
    BestRGB =  [[0.6326696 , 0.2045558 , 0.1269946],[0.2284569 , 0.7373523 , 0.0341908],[0.0000000 , 0.0095142 , 0.8156958]]
    BestRGB=np.array(BestRGB)
    #Beta RGB (D50)
    BetaRGB =  [[0.6712537 , 0.1745834 , 0.1183829],[0.3032726 , 0.6637861 , 0.0329413],[0.0000000 , 0.0407010 , 0.7845090]]
    BetaRGB=np.array(BetaRGB)
    # Bruce RGB (D65)
    BruceRGB = [[0.4674162 , 0.2944512 , 0.1886026],[0.2410115 , 0.6835475 , 0.0754410],[0.0219101 , 0.0736128 , 0.9933071]]
    BruceRGB=np.array(BruceRGB)
    #CIE RGB (E)(D55)
    CIERGB =   [[0.4887180 , 0.3106803 , 0.2006017],[0.1762044 , 0.8129847 , 0.0108109],[0.0000000 , 0.0102048 , 0.9897952]]
    CIERGB=np.array(CIERGB)
    #ColorMatch RGB (D50)
    CmatchRGB =[[0.5093439 , 0.3209071 , 0.1339691],[0.2748840 , 0.6581315 , 0.0669845],[0.0242545 , 0.1087821 , 0.6921735]]
    CmatchRGB=np.array(CmatchRGB)
    # Don RGB4 (D50)
    DonRGB4 =  [[0.6457711 , 0.1933511 , 0.1250978],[0.2783496 , 0.6879702 , 0.0336802],[0.0037113 , 0.0179861 , 0.8035125]]
    DonRGB4=np.array(DonRGB4)
    # ECI RGB (D50)
    ECIRGB =   [[0.6502043 , 0.1780774 , 0.1359384],[0.3202499 , 0.6020711 , 0.0776791],[0.0000000 , 0.0678390 , 0.7573710]]
    ECIRGB=np.array(ECIRGB)
    #Ekta Space PS5(D50)
    ESPS5 =    [[0.5938914 , 0.2729801 , 0.0973485],[0.2606286 , 0.7349465 , 0.0044249],[0.0000000 , 0.0419969 , 0.7832131]]
    ESPS5=np.array(ESPS5)
    # NTSC RGB (C)
    NTSCRGB =  [[0.6068909 , 0.1735011 , 0.2003480],[0.2989164 , 0.5865990 , 0.1144845],[0.0000000 , 0.0660957, 1.1162243]]
    NTSCRGB=np.array(NTSCRGB)
    # Pal/secam RGB (D65)
    PSRGB =    [[0.4306190 , 0.3415419 , 0.1783091],[0.2220379 , 0.7066384 , 0.0713236],[0.0201853 , 0.1295504 , 0.9390944]]
    PSRGB=np.array(PSRGB)
    # Prophoto RGB (D50)
    PPRGB =    [[0.7976749 , 0.1351917 , 0.0313534],[0.2880402 , 0.7118741 , 0.0000857],[0.0000000 , 0.0000000 , 0.8252100]]
    PPRGB=np.array(PPRGB)
    #SMPTE-C RGB (D65)
    SMPT_CRGB =[[0.3935891 , 0.3652497 , 0.1916313],[0.2124132 , 0.7010437 , 0.0865432],[0.0187423 , 0.1119313 , 0.9581563]]
    SMPT_CRGB=np.array(SMPT_CRGB)
    #sRGB (With D65)
    sRGB =        [[0.412453 , 0.357580 , 0.180423],[0.212671 , 0.715160 , 0.072169],[0.019334 , 0.119193 , 0.950227]]
    sRGB=np.array(sRGB)
    #Wide Gamut RGB (use with D50)
    WGRGB =    [[0.7161046 , 0.1009296 , 0.1471858],[0.2581874 , 0.7249378 , 0.0168748],[0.0000000 , 0.0517813 , 0.7734287]]
    WGRGB=np.array(WGRGB)