# -*- coding: utf-8 -*-


import lxml.etree
import lxml.builder

#tree = etree.parse("data.xml")
#for user in tree.xpath("/users/user/nom"):
#    print(user.text)
    
#tree = etree.parse("data.xml")
#for user in tree.xpath("/users/user"):
#    print(user.get("data-id"))
#    
#tree = etree.parse("data.xml")
#for user in tree.xpath("/users/user[metier='Veterinaire']/nom"):
#    print(user.text)
    
#users = etree.Element("users")
#user = etree.SubElement(users, "user")
#user.set("data-id","101")
#nom = etree.SubElement(user, "nom")
#nom.text = "Zorro"
#metier =etree.SubElement(user,"metier")
#metier.text = "Sado"
#print(etree.tostring(users, pretty_print=True))

#tree = etree.ElementTree(users)
#tree.write("test.xml")

#creation arbre xml
E = lxml.builder.ElementMaker()
ROOT = E.root
KEYPOINTS = E.keypoints
KP1 = E.kp1
KP2 = E.kp2

the_doc = ROOT(
        KEYPOINTS(
            KP1('keypoint1 coords', name='kp1'),
            KP2('keypoint2 coords', name='kp2'),
            )
        )
test = lxml.etree.tostring(the_doc, pretty_print=True)        


# creation fichier xml
fo = open("foo.xml", "wb")
fo.write(test);

# Close opend file
fo.close()