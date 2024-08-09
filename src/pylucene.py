import os
import lucene
import pandas as pd
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.index import DirectoryReader, IndexWriter, IndexWriterConfig
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.search import BooleanQuery

from src import variables
from src.utils import getDocDict



class PyLuceneWrapper:

    def __init__(self, recreated_index=False, documents=None ):
        # Initialize the JVM
        self.env = lucene.initVM()
        # Initialize the analyzer
        self.analyzer = StandardAnalyzer()
        # Initialize the index directory
        self.index_dir = None
        # set max clause count
        BooleanQuery.setMaxClauseCount(2048*2)

        if recreated_index or not os.path.exists(self.indexpath()):
            self.create_index(documents)
        else:
            self.index_dir = FSDirectory.open(Paths.get(self.indexpath()))
    def create_index(self, documents):
        #     remove the index directory
        if os.path.exists(self.indexpath()):
            for file in os.listdir(self.indexpath()):
                os.remove(os.path.join(self.indexpath(), file))
            os.rmdir(self.indexpath())
        #     create the index directory
        os.makedirs(self.indexpath())
        self.index_dir = FSDirectory.open(Paths.get(self.indexpath()))


        # Initialize the index writer
        config = IndexWriterConfig(self.analyzer)
        writer = IndexWriter(self.index_dir, config)
        # Add documents to the index
        for title, text in documents.items():
            doc = Document()
            doc.add(TextField("title", title, Field.Store.YES))
            doc.add(TextField("content", text, Field.Store.YES))
            writer.addDocument(doc)
        writer.close()

    def indexpath(self):
        index_path = "index"
        # if wd ends with src, remove src
        if os.getcwd().endswith("src"):
            index_path = "../" + index_path
        return index_path

    def search_index(self, query_string, num_results=10):
        searcher = IndexSearcher(DirectoryReader.open(self.index_dir))
        query_parser = QueryParser("content", self.analyzer)
        query = query_parser.parse(query_string)

        hits = searcher.search(query, num_results).scoreDocs
        results = []
        for hit in hits:
            doc_id = hit.doc
            doc = searcher.doc(doc_id)
            results.append((doc.get("title"), hit.score))
        return results

if __name__ == '__main__':

    doc_dict = getDocDict(filepath_video_games=variables.filepath_video_games, csv_doc_dict=variables.csv_doc_dict)

    lucene_wrapper = PyLuceneWrapper(documents=doc_dict)


    query = "handheld game consol produc nintendo fourth system nintendo 3d famili handheld consol follow origin nintendo 3d nintendo 3d xl nintendo 2d system releas japan octob 11 2014 australia new zealand novemb 21 2014 januari 6 2015 europ special club nintendoexclus ambassador edit retail europ februari 13 2015 like origin 3d new nintendo 3d also larger variant new nintendo 3d xl releas three region north america new nintendo 3d xl releas februari 13 2015 standards new nintendo 3d releas later septemb 25 2015 improv upon previou model includ upgrad processor increas ram analog point stick cstick two addit shoulder trigger zr zl face detect optim autostereoscop 3d display includ 4 gb microsd card builtin nfc well minor design chang color face button avail face plate smallers model new nintendo 3d receiv posit review critic although critic certain aspect design microsd slot placement consol prais improv perform addit control option better 3d imag qualiti juli 2017 leadup releas new nintendo 2d xl nintendo confirm product standards new nintendo 3d japan end xl model remain product juli 2019 product ceas remov websit hardwar new nintendo 3d famili featur variou chang prior model system featur slightli refin design featur color face button resembl super famicom pal version super nintendo entertain system color scheme new nintendo 3dss screen 12 time size origin nintendo 3d screen xl variant size predecessor model produc ip screen upper display still retain old tn screen upper display known correl model number product date display type nintendo also publicli address discrep product new featur known super stabl 3d improv qualiti system autostereoscop 3d effect use sensor detect angl player view screen adjust effect compens sensor also use ambient light sensor automat bright adjust system bodi slightli larger previou iter xl variant weigh slightli less previou 3d xl system game card slot stylu holder power button reloc base hardwar wireless switch also replac softwar toggl standard new nintendo 3d featur interchang front back plate 38 differ design avail launch japan xl variant allow use plate instead coupl fix metal design intern specif devic also updat includ addit processor core increas 256 mb ram near field commun support use amiibo product control new system expand inclus point stick right hand side devic refer cstick addit zl zr shoulder button allow function equival circl pad pro addon peripher releas previou model addit button backwardscompat game program use circl pad pro unlik previou model use standard sd card new nintendo 3d line use microsd card data storag store alongsid batteri behind devic rear cover need screw remov order access microsd card slot data also transfer sd card wirelessli use system smb client access like pc new system continu use ac adapt dsi dsi xl devic 3d famili like nintendo 3d xl japan europ first time north america ac adapt includ consol must obtain separ softwar servic asid minor adjust reflect hardwar design differ system softwar new nintendo 3d otherwis ident origin 3d offer onlin featur nintendo network multiplay onlin game nintendo eshop download purchas game streetpass spotpass web browser updat includ html5base video playback support japanes model content filter activ default disabl registr credit card intend prevent children visit matur websit dsi dsi xl previou 3d model new nintendo 3d famili remain compat game releas ds 3d exclud game use game boy advanc cartridg slot 3d game improv perform andor graphic new system due upgrad hardwar cstick zlzr control backward compat game support circl pad pro addon game xenoblad chronicl 3d specif optim upgrad hardwar exclus new nintendo 3d support prior model march 2016 nintendo began releas sne titl virtual consol new 3d support perfect pixel mode allow game play pillar box squar pixel rather origin 43 proport like previou model 3d game download softwar regionlock ds cartridg remain regionfre due differ size peripher design fit shape origin nintendo 3d use new system game data transfer previou 3d system new system either manual wireless though data new system transfer older system april 13 2015 uniti technolog announc uniti engin would support new nintendo 3d releas new nintendo 3d first announc japanes nintendo direct present stream august 29 2014 new nintendo 3d 3d releas japan octob 11 2014 regulars version made avail black whitecolor version made avail metal blue metal black version addit limit edit design 38 differ face plate design avail launch japan showcas prelaunch televis commerci featur jpop perform kyari pamyu pamyu 230000 unit sold first two day avail new nintendo 3d xl variant first releas outsid japanin australia new zealand novemb 21 2014 smaller model avail white europ new nintendo 3d first made avail onlin januari 6 2015 special white ambassador edit bundl exclus club nintendo member charg dock two face plate includ januari 14 2015 nintendo announc new system would releas retail north america europ februari 13 2015 europ new nintendo 3d avail black white xl variant metal black metal blue north america xl model releas metal red metal black renam new red new black special monster hunter 4 ultimatethem variant also releas launch region 335000 unit sold first week avail europ north america xl model origin releas north america although nintendo rule possibl releas regular new nintendo 3d futur nintendo america repres damon baker explain want confus consum face plate enough reason smallers system releas north america social media campaign emerg call upon nintendo america releas model north america march 2015 fcc lift inform embargo regard regul detail perform septemb 2014 standard new nintendo 3d model suggest nintendo america inde consid releas smaller standard model one point august 31 2015 gamestop manag confer la vega nintendo america confirm standard new 3d system would launch region septemb 25 2015 theme bundl includ consol game softwar two facepl amiibo card second legend zeldathem xl bundl hyrul edit also announc gamestop exclus releas octob 30 2015 januari 2016 special pokmonthem new nintendo 3d bundl announc releas north america februari 27 2016 coincid 20th anniversari virtual consol releas origin pokmon game model bundl pokmon red pokmon blue charizard blastoisethem facepl download home menu skin august 2016 super mario 3d land new 3d bundl two facepl releas north america exclus target walmart nintendo releas black whitecolor new 3d model mariothem design north america novemb 2016 black friday two model sold us9999a price 20 higher 2d juli 2017 nintendo confirm leadup releas new nintendo 2d xl product standards new nintendo 3d japan end xl model halt product juli 2019 recept review new nintendo 3d line posit critic felt new superst 3d system success improv consist view angl devic stereoscop 3d effect especi game requir use gyroscop ign writer note constant sway occasion jolt morn train commut occasion shatter new system stereoscop spell even system quickli adjust snap back focu improv technic specif new system also note make devic oper system respons provid modest perform enhanc exist game monster hunter 4 ultim incorpor circl pad pro addit shoulder button secondari analog stick devic prais along potenti use port game home consol opinion mix design cstick howev gamespot felt surprisingli respons ign drew comparison similar point stick sometim found thinkpad laptop felt good occasion function camera control aim thirdperson game would function well intens use case firstperson shooter due size lack grip comparison circl pad aspect devic design note ign felt face plate option regulars model ad level person consol face plate take younger gamer particular accessori could easili end bargain bin faster say limit edit perfect dark zero xbox 360 face plate nintendo decis exclud featur xl version also consid odd wire felt new locat power button card slot stylu holder inconveni critic also felt switch microsd card reloc sd card slot batteri compart would make manual transfer data previou 3d model trickier gamespot lament difficulti unscrew rear cover xl cite stubborn screw panel practic refus detach nintendo decis bundl ac adapt new model critic particularli case firsttim 3d owner gamespot felt new nintendo 3d xl best handheld nintendo ever made recommend firsttim 3d owner regard exist 3d owner new system recommend show interest exclus want better overal experi give consol 88 10 ign conclud addit control increas process power set system nice futur your late 3d parti youv got back catalogu featur best handheld game recent year best game full stop decemb 31 2016 994 million unit new nintendo 3d new nintendo 3d xl ship worldwid"
    results = lucene_wrapper.search_index(query)

    for title, score in results:
        print(f"Title: {title}, Score: {score}")
