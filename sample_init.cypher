// ── Constraints ──────────────────────────────────────────────
CREATE CONSTRAINT actor_name IF NOT EXISTS
FOR (a:Actor) REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT movie_title IF NOT EXISTS
FOR (m:Movie) REQUIRE m.title IS UNIQUE;

CREATE CONSTRAINT director_name IF NOT EXISTS
FOR (d:Director) REQUIRE d.name IS UNIQUE;

// ── Indexes ─────────────────────────────────────────────────
CREATE INDEX movie_released IF NOT EXISTS
FOR (m:Movie) ON (m.released);

// ── Seed data ───────────────────────────────────────────────
MERGE (tom:Actor {name: "Tom Hanks"})
SET tom.born = 1956;

MERGE (meg:Actor {name: "Meg Ryan"})
SET meg.born = 1961;

MERGE (forrest:Movie {title: "Forrest Gump"})
SET forrest.released = 1994, forrest.tagline = "Life is like a box of chocolates.";

MERGE (youve:Movie {title: "You've Got Mail"})
SET youve.released = 1998;

MERGE (castaway:Movie {title: "Cast Away"})
SET castaway.released = 2000;

MERGE (rob:Director {name: "Robert Zemeckis"});

// ── Relationships ───────────────────────────────────────────
MATCH (tom:Actor {name: "Tom Hanks"}), (forrest:Movie {title: "Forrest Gump"})
MERGE (tom)-[:ACTED_IN {role: "Forrest"}]->(forrest);

MATCH (tom:Actor {name: "Tom Hanks"}), (youve:Movie {title: "You've Got Mail"})
MERGE (tom)-[:ACTED_IN {role: "Joe Fox"}]->(youve);

MATCH (tom:Actor {name: "Tom Hanks"}), (castaway:Movie {title: "Cast Away"})
MERGE (tom)-[:ACTED_IN {role: "Chuck Noland"}]->(castaway);

MATCH (meg:Actor {name: "Meg Ryan"}), (youve:Movie {title: "You've Got Mail"})
MERGE (meg)-[:ACTED_IN {role: "Kathleen Kelly"}]->(youve);

MATCH (rob:Director {name: "Robert Zemeckis"}), (forrest:Movie {title: "Forrest Gump"})
MERGE (rob)-[:DIRECTED]->(forrest);

MATCH (rob:Director {name: "Robert Zemeckis"}), (castaway:Movie {title: "Cast Away"})
MERGE (rob)-[:DIRECTED]->(castaway);
