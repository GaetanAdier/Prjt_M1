# -*- coding: utf-8 -*-
"""
Ce module sert a la génération de documentation automatique avec Sphinx. Il se compose de deux fonction : configure_doc() et genere_doc() qui devront toutes deux être apellées comme expliqué ci dessous.
Pour créer la documentation d'un module, il faudra mettre le fichier contenant ces deux fonctions dans le même dossier que les fichiers sources, et taper les lignes de code suivantes dans les import de votre module de base.
::
    from sphinx_doc import genere_doc
    from sphinx_doc import configure_doc

Il faudra ensuite dans votre main apeller les fonction commes expliqué dans leur documetation respective.
Le résultat de la génération de vos documentation devra vous donner une arborescence semblable a celle décrite ci dessous : 

.. image:: arbo.png 
   :align: center

.. note:: La documentation générée dépendra des commentaires placés sur chaque fonction. Pour connaitre les syntaxe à appliquer, se référer a la documentation suivante : 
          Tutoriel Openclassroom <http://deusyss.developpez.com/tutoriels/Python/SphinxDoc/#LIII-D/>

"""

import sphinx
import sphinx.quickstart
import sphinx.apidoc


def configure_doc():
    '''
            
    
    Procédure de configuration de l'auto-configuration
    
    Il y a a l'appel de cette procedure une certain quantité de parametres qui seront a completer \:

     * **Root path for the documentation** : On renseignera le nom "doc" qui créera un dossier du même nom pour stocker la documentation 
     * **Separate source and build directories (Y/N) :** Ici on renseignera 'n' (par défaut)
     * **Name prefix for templates and static dir :** Laisser par defaut ("_") 
     * **Project name :** Nom du projet (au choix)
     * **Author name(s) :** Nom de l'auteur (au choix)
     * **Project version :** Numéro de version du projet
     * **Project release :** Numéro de release
     * **Source file suffix :** Laisser par defaut (".rst")
     * **Name of your master document (without suffix) :** Laisser par defaut ("index.rst")
     * **Do you want to use the epub builder (y/N) :** Mettre a non (je ne sait pas ce que c'est)
     * **autodoc: automatically insert docstrings from modules (y/N) :** Mettre à oui (sinon on ne pourra pas utiliser l'autodocumentation)
     * **doctest: automatically test code snippets in doctest blocks (y/N) :** Laisser a non (je sait pas ce que c'est non plus)
     * **intersphinx: link between Sphinx documentation of different projects (y/N) :** Dans notre cas mettre a non
     * **todo: write « todo » entries that can be shown or hidden on build (y/N) :** Cela peut être bien de le mettre à oui
     * **coverage: checks for documentation coverage (y/N) :** je sait pas ce que c'est mais dans le doute on laisse a non
     * **pngmath: include math, rendered as PNG images (y/N) :** Mettre à oui pour pouvoir inclure des formules au format LateX
     * **mathjax: include math, rendered in the browser by MathJax (y/N) :** Substitut a pngmath, mettre a non
     * **ifconfig: conditional inclusion of content based on config values (y/N) :** Je sait pas mais quand on le laisse a "non" sa fonctionne très bien alors ^^
     * **viewcode: include links to the source code of documented Python objects (y/N) :** Fait le lien avec le code source, intéressant a mettre a oui
     * **Create Makefile? (Y/n) :** Mettre a oui
     * **Create Windows command file? (Y/n) :** Mettre a oui
    

        
    
    
    
    .. note:: Cette fonction n'est apellée qu'une fois a la création du projet de documentation
    
        

    
    '''
    
    sphinx.quickstart.main(['sphinx-quickstart'])


def genere_doc():
    """
    Fonction de génération de l'autodocumentation. La documentation sera générée a chaque appel de cette procédure.
    
   
    
    
    .. warning:: Cette fonction peut faire ressortir certain warning sans raison apparente. Cela n'influe pas sur le résultat de la documentation qui sera tout de même générée mais il sera nécessaire de couper l'éxécution de la console de manière manuelle afin d'effectuer une nouvelle éxécution (appuyer sur le bouton en forme e fleche verte en haut a droite de la console)

    """
    sphinx.apidoc.main(['sphinx-apidoc', '-f', '--output-dir=doc/generated', './'])
    sphinx.main(['sphinx-build', '-b', 'html', 'doc', 'doc/_build/html'])
#    sphinx.main(['sphinx-build', '-b', 'latex', 'doc', 'doc/_build/latex'])