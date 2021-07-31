'''
Created on 30/07/2021

@author: yolan
'''

if __name__ == '__main__':
    pass

def preprocessBank(infile,outfile):
    stext1 = 'no'
    stext2 = 'yes'

    rtext1 = '0'
    rtext2 = '1'

    fid = open(infile,"r")
    oid = open(outfile,"w")
    for s in fid:
        if s.find(stext1)>-1:
            oid.write(s.replace(stext1, rtext1))
        elif s.find(stext2)>-1:
                oid.write(s.replace(stext2, rtext2))
        
    fid.close()
    oid.close()