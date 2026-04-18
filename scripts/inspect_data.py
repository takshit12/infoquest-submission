"""Sample real data and compute data-quality stats to inform embedding/ranking design."""
import os
import json
import psycopg2
import psycopg2.extras

DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://developer:devread2024@34.79.32.228:5432/candidate_profiles",
)

def run():
    conn = psycopg2.connect(DSN, connect_timeout=15)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def section(title):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    section("CANDIDATE NULL RATES & SIMPLE STATS")
    cur.execute(
        """
        SELECT
          COUNT(*) AS total,
          COUNT(headline) AS n_headline,
          COUNT(years_of_experience) AS n_yoe,
          COUNT(city_id) AS n_city,
          COUNT(nationality_id) AS n_nationality,
          COUNT(phone) AS n_phone,
          COUNT(date_of_birth) AS n_dob,
          COUNT(gender) AS n_gender,
          MIN(years_of_experience) AS yoe_min,
          MAX(years_of_experience) AS yoe_max,
          ROUND(AVG(years_of_experience)::numeric, 1) AS yoe_avg,
          MIN(LENGTH(headline)) AS headline_min_len,
          MAX(LENGTH(headline)) AS headline_max_len,
          ROUND(AVG(LENGTH(headline))::numeric, 1) AS headline_avg_len
        FROM candidates
        """
    )
    print(json.dumps(cur.fetchone(), default=str, indent=2))

    section("WORK EXPERIENCE STATS (description is the main prose field)")
    cur.execute(
        """
        SELECT
          COUNT(*) AS total_rows,
          COUNT(description) AS n_desc,
          COUNT(*) FILTER (WHERE is_current) AS n_current,
          MIN(LENGTH(description)) AS desc_min_len,
          MAX(LENGTH(description)) AS desc_max_len,
          ROUND(AVG(LENGTH(description))::numeric, 0) AS desc_avg_len,
          COUNT(DISTINCT candidate_id) AS candidates_with_work
        FROM work_experience
        """
    )
    print(json.dumps(cur.fetchone(), default=str, indent=2))

    section("WORK EXPERIENCE per-candidate distribution")
    cur.execute(
        """
        SELECT per_cand, COUNT(*) AS n_candidates
        FROM (
          SELECT candidate_id, COUNT(*) AS per_cand
          FROM work_experience GROUP BY candidate_id
        ) x
        GROUP BY per_cand ORDER BY per_cand
        """
    )
    for r in cur.fetchall():
        print(f"  {r['per_cand']} roles: {r['n_candidates']} candidates")

    section("SKILLS per candidate distribution")
    cur.execute(
        """
        SELECT per_cand, COUNT(*) AS n_candidates
        FROM (
          SELECT candidate_id, COUNT(*) AS per_cand
          FROM candidate_skills GROUP BY candidate_id
        ) x
        GROUP BY per_cand ORDER BY per_cand
        """
    )
    for r in cur.fetchall():
        print(f"  {r['per_cand']} skills: {r['n_candidates']} candidates")

    section("TOP 20 INDUSTRIES (by candidate count via work experience)")
    cur.execute(
        """
        SELECT c.industry, COUNT(DISTINCT we.candidate_id) AS n_candidates
        FROM work_experience we
        JOIN companies c ON c.id = we.company_id
        WHERE c.industry IS NOT NULL
        GROUP BY c.industry
        ORDER BY n_candidates DESC
        LIMIT 20
        """
    )
    for r in cur.fetchall():
        print(f"  {r['n_candidates']:>5}  {r['industry']}")

    section("TOP 20 COUNTRIES by candidate count")
    cur.execute(
        """
        SELECT co.name AS country, COUNT(*) AS n
        FROM candidates ca
        JOIN cities ci ON ci.id = ca.city_id
        JOIN countries co ON co.id = ci.country_id
        GROUP BY co.name ORDER BY n DESC LIMIT 20
        """
    )
    for r in cur.fetchall():
        print(f"  {r['n']:>5}  {r['country']}")

    section("TOP 20 JOB TITLES")
    cur.execute(
        """
        SELECT job_title, COUNT(*) AS n
        FROM work_experience GROUP BY job_title ORDER BY n DESC LIMIT 20
        """
    )
    for r in cur.fetchall():
        print(f"  {r['n']:>5}  {r['job_title']}")

    section("TOP 20 SKILLS")
    cur.execute(
        """
        SELECT s.name, COUNT(*) AS n
        FROM candidate_skills cs
        JOIN skills s ON s.id = cs.skill_id
        GROUP BY s.name ORDER BY n DESC LIMIT 20
        """
    )
    for r in cur.fetchall():
        print(f"  {r['n']:>5}  {r['name']}")

    section("SKILL CATEGORIES (top-level, with counts)")
    cur.execute(
        """
        SELECT sc.name, COUNT(s.id) AS n_skills
        FROM skill_categories sc
        LEFT JOIN skills s ON s.category_id = sc.id
        WHERE sc.parent_id IS NULL
        GROUP BY sc.name ORDER BY n_skills DESC
        """
    )
    for r in cur.fetchall():
        print(f"  {r['n_skills']:>4}  {r['name']}")

    section("PROFICIENCY LEVELS (ranked)")
    cur.execute("SELECT name, rank FROM proficiency_levels ORDER BY rank")
    for r in cur.fetchall():
        print(f"  rank={r['rank']}  {r['name']}")

    section("SAMPLE: 3 FULL CANDIDATE PROFILES (assembled)")
    cur.execute(
        """
        SELECT id FROM candidates
        WHERE headline IS NOT NULL AND years_of_experience IS NOT NULL
        ORDER BY years_of_experience DESC
        LIMIT 3
        """
    )
    sample_ids = [r["id"] for r in cur.fetchall()]
    for cid in sample_ids:
        cur.execute(
            """
            SELECT ca.id, ca.first_name, ca.last_name, ca.headline,
                   ca.years_of_experience, ca.date_of_birth, ca.gender,
                   ci.name AS city, co.name AS country,
                   nat.name AS nationality
            FROM candidates ca
            LEFT JOIN cities ci ON ci.id = ca.city_id
            LEFT JOIN countries co ON co.id = ci.country_id
            LEFT JOIN countries nat ON nat.id = ca.nationality_id
            WHERE ca.id = %s
            """,
            (cid,),
        )
        cand = cur.fetchone()
        print(f"\n--- {cand['first_name']} {cand['last_name']} ({cand['city']}, {cand['country']}) ---")
        print(f"  Headline: {cand['headline']}")
        print(f"  YoE: {cand['years_of_experience']} | Nationality: {cand['nationality']}")

        cur.execute(
            """
            SELECT we.job_title, we.start_date, we.end_date, we.is_current, we.description,
                   co.name AS company, co.industry
            FROM work_experience we
            JOIN companies co ON co.id = we.company_id
            WHERE we.candidate_id = %s
            ORDER BY we.start_date DESC
            """,
            (cid,),
        )
        for w in cur.fetchall():
            end = "present" if w["is_current"] else str(w["end_date"])
            desc_preview = (w["description"] or "").strip()
            if len(desc_preview) > 180:
                desc_preview = desc_preview[:180] + "…"
            print(f"  Work: {w['job_title']} @ {w['company']} [{w['industry']}] ({w['start_date']}–{end})")
            if desc_preview:
                print(f"        {desc_preview}")

        cur.execute(
            """
            SELECT d.name AS degree, f.name AS field, i.name AS inst,
                   e.start_year, e.graduation_year
            FROM education e
            JOIN degrees d ON d.id = e.degree_id
            JOIN fields_of_study f ON f.id = e.field_of_study_id
            JOIN institutions i ON i.id = e.institution_id
            WHERE e.candidate_id = %s
            ORDER BY e.graduation_year DESC NULLS LAST
            """,
            (cid,),
        )
        for e in cur.fetchall():
            print(f"  Edu : {e['degree']} in {e['field']}, {e['inst']} ({e['start_year']}–{e['graduation_year']})")

        cur.execute(
            """
            SELECT s.name, cs.years_of_experience, cs.proficiency_level,
                   sc.name AS category
            FROM candidate_skills cs
            JOIN skills s ON s.id = cs.skill_id
            JOIN skill_categories sc ON sc.id = s.category_id
            WHERE cs.candidate_id = %s
            ORDER BY cs.years_of_experience DESC NULLS LAST
            LIMIT 10
            """,
            (cid,),
        )
        sk = cur.fetchall()
        if sk:
            print("  Top skills:")
            for s in sk:
                print(f"    - {s['name']} [{s['category']}] (yoe={s['years_of_experience']}, {s['proficiency_level']})")

        cur.execute(
            """
            SELECT l.name, p.name AS proficiency, p.rank
            FROM candidate_languages cl
            JOIN languages l ON l.id = cl.language_id
            JOIN proficiency_levels p ON p.id = cl.proficiency_level_id
            WHERE cl.candidate_id = %s
            ORDER BY p.rank DESC
            """,
            (cid,),
        )
        langs = cur.fetchall()
        if langs:
            print("  Languages: " + ", ".join(f"{l['name']} ({l['proficiency']})" for l in langs))

    conn.close()

if __name__ == "__main__":
    run()
