import sqlite3
import networkx as nx
import json
import os

class MemoryManager:
    def __init__(self, db_path=None):
        if db_path is None:
            self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "databases", "memory.db")
        else:
            self.db_path = db_path
        self.graph = nx.DiGraph()
        self.setup_db()
        self.load_graph()

    def setup_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table to store individual hand sign detection events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sign TEXT NOT NULL,
                confidence REAL,
                drafter_guess TEXT,
                critic_approved BOOLEAN
            )
        ''')
        
        # Table to store transition counts (edges in the NetworkX graph)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sign_transitions (
                source_sign TEXT,
                target_sign TEXT,
                weight INTEGER DEFAULT 1,
                PRIMARY KEY (source_sign, target_sign)
            )
        ''')
        
        # Table to store geometric templates for Cosine Distance verification
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sign_templates (
                sign TEXT PRIMARY KEY,
                landmarks TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def load_graph(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT source_sign, target_sign, weight FROM sign_transitions")
        for row in cursor.fetchall():
            self.graph.add_edge(row[0], row[1], weight=row[2])
        conn.close()

    def get_last_sign(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sign FROM detection_history ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def get_likely_next_signs(self, current_sign, top_n=3):
        if current_sign not in self.graph:
            return []
        
        # Get successors sorted by weight
        successors = self.graph.successors(current_sign)
        edges = [(succ, self.graph[current_sign][succ]['weight']) for succ in successors]
        edges.sort(key=lambda x: x[1], reverse=True)
        return [edge[0] for edge in edges[:top_n]]

    def record_detection(self, sign, confidence, drafter_guess, critic_approved):
        last_sign = self.get_last_sign()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert detection
        cursor.execute('''
            INSERT INTO detection_history (sign, confidence, drafter_guess, critic_approved)
            VALUES (?, ?, ?, ?)
        ''', (sign, confidence, drafter_guess, critic_approved))
        
        # Update transition graph if we have a previous sign
        if last_sign and last_sign != sign:
            cursor.execute('''
                INSERT INTO sign_transitions (source_sign, target_sign, weight)
                VALUES (?, ?, 1)
                ON CONFLICT(source_sign, target_sign) DO UPDATE SET weight = weight + 1
            ''', (last_sign, sign))
            
            # Update local NetworkX graph
            if self.graph.has_edge(last_sign, sign):
                self.graph[last_sign][sign]['weight'] += 1
            else:
                self.graph.add_edge(last_sign, sign, weight=1)
                
        conn.commit()
        conn.close()

    def get_template(self, sign):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT landmarks FROM sign_templates WHERE sign = ?", (sign,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None

    def save_template(self, sign, landmarks_vector):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sign_templates (sign, landmarks)
            VALUES (?, ?)
            ON CONFLICT(sign) DO UPDATE SET landmarks = excluded.landmarks
        ''', (sign, json.dumps(landmarks_vector)))
        conn.commit()
        conn.close()
