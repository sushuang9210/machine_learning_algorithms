import numpy
import pymongo
import pprint
import sqlite3
import requests
import cx_Oracle
import sys
import mysql.connector as mc
import urllib.parse
from pymongo import MongoClient
from riko.modules import fetch, fetchsitefeed

class Mapper:
    def __init__(self,config_file):
        self.config_file = config_file
        with open(config_file) as f:
            content = f.readlines()
        self.config = [x.strip() for x in content]
        file_type_string = self.config[0].split(' ')
        self.file_type = file_type_string[0]

    def convert_data(self,data,data_type):
        if data_type == '0':
            return int(float(data))
        if data_type == '1':
            return float(data)
        if data_type == '2':
            return data

    def get_file_data(self):
        in_file = self.config[3].split(' ')[0]
        general_data = []
        with open(in_file) as f:
            content = f.readlines()
            content_data = [x.strip() for x in content]
            for i in range(len(content_data)-1):
                feature_i = []
                data = content_data[i+1].split(' ')
                for j in range(self.num_feature):
                    feature_i.append(self.convert_data(data[j],self.data_type[j]))
                general_data.append(feature_i)
        return numpy.array(general_data)

    def get_db_data(self):
        general_data = []
        dbname = self.config[3].split(' ')[0]
        dbip = self.config[4].split(' ')[0]
        dbport = int(self.config[5].split(' ')[0])
        collectioname = self.config[6].split(' ')[0]
        keyname = self.config[7].split(' ')
        use_username = int(self.config[8].split(' ')[0])
        #print(use_username)
        if use_username==1:
            username = urllib.parse.quote_plus(self.config[9].split(' ')[0])
            password = urllib.parse.quote_plus(self.config[10].split(' ')[0])
            client = MongoClient('mongodb://%s:%s@%s' % (username, password, dbip))
        else:
            client = MongoClient(dbip, dbport)
        db = client[dbname]
        collection = db[collectioname]
        for item in collection.find():
            item_info = []
            for i in range(self.num_feature):
                picked_item = item
                keyname_i = keyname[i]
                keys = keyname_i.split(',')
                for key in keys:
                    #print(key)
                    picked_item = picked_item[key]
                item_info.append(self.convert_data(picked_item,self.data_type[i]))
            general_data.append(item_info)
        return numpy.array(general_data)

    def get_sql_data(self):
        general_data = []
        dbname = self.config[3].split(' ')[0]
        tablename = self.config[4].split(' ')[0]
        connection = sqlite3.connect(dbname)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM "+ tablename)
        result = cursor.fetchall()
        for r in result:
            item_info = []
            for i in range(len(r)):
                item_info.append(self.convert_data(r[i],self.data_type[i]))
            general_data.append(item_info)
        cursor.close()
        connection.close()
        return numpy.array(general_data)

    def get_mysql_data(self):
        general_data = []
        dbname = self.config[3].split(' ')[0]
        dbip = self.config[4].split(' ')[0]
        tablename = self.config[5].split(' ')[0]
        username = self.config[6].split(' ')[0]
        password = self.config[7].split(' ')[0]
        connection = mc.connect (host = dbip,user = username,passwd = password,db = dbname)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM "+ tablename)
        result = cursor.fetchall()
        for r in result:
            item_info = []
            for i in range(len(r)):
                item_info.append(self.convert_data(r[i],self.data_type[i]))
            general_data.append(item_info)
        cursor.close()
        connection.close()
        return numpy.array(general_data)

    def get_oracle_data(self):
        general_data = []
        dbname = self.config[3].split(' ')[0]
        dbip = self.config[4].split(' ')[0]
        dbport = self.config[5].split(' ')[0]
        tablename = self.config[6].split(' ')[0]
        username = self.config[7].split(' ')[0]
        password = self.config[8].split(' ')[0]
        oraclepath = self.config[9].split(' ')[0]
        sys.path.append('/opt/instantclient_12_2/')
        dsn = cx_Oracle.makedsn(dbip, int(dbport), service_name=dbname)
        connection = cx_Oracle.connect(username,password,dsn)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM "+ tablename)
        result = cursor.fetchall()
        for r in result:
            item_info = []
            for i in range(len(r)):
                item_info.append(self.convert_data(r[i],self.data_type[i]))
            general_data.append(item_info)
        cursor.close()
        connection.close()
        return numpy.array(general_data)

    def get_stream_data(self):
        general_data = []
        website = self.config[3].split(' ')[0]
        keyname = self.config[4].split(' ')
        num_data = int(self.config[5].split(' ')[0])
        stream = fetch.pipe(conf={'url': website})

        for i in range(num_data):
            item = next(stream)
            item_info = []
            for i in range(self.num_feature):
                key = keyname[i]
                picked_item = item[key]
                item_info.append(self.convert_data(picked_item,self.data_type[i]))
            general_data.append(item_info)
        return numpy.array(general_data)

    def get_general_data(self):
        general_data = []
        if self.file_type == '0':
            general_data.append(in_file)
        else:
            self.num_feature = int(self.config[1].split(' ')[0])
            self.data_type = self.config[2].split(' ')
        if self.file_type == '1':
            general_data = self.get_file_data()
        if self.file_type == '2':
            general_data = self.get_db_data()
        if self.file_type == '3':
            general_data = self.get_sql_data()
        if self.file_type == '4':
            general_data = self.get_mysql_data()
        if self.file_type == '5':
            general_data = self.get_oracle_data()
        if self.file_type == '6':
            general_data = self.get_stream_data()
        return general_data
