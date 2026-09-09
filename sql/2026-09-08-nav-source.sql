-- Adds the column the nav-source edge function feeds.
--
-- `referrer` being null only says no referrer reached the page. `nav_source`
-- holds the document request's Sec-Fetch-Site, which distinguishes:
--   none        typed, bookmarked, opened with no context  (true direct)
--   cross-site  another site linked here and stripped the referrer
--   same-site   a subdomain
--   same-origin in-site navigation
--   unknown     client sent no Sec-Fetch-Site header
--
-- Safe to run against the live table: nullable, no default, no rewrite.
-- Existing rows stay null and render as "(not recorded)" on /analytics.

alter table public.page_views
  add column if not exists nav_source text;

alter table public.page_views
  add constraint page_views_nav_source_check
  check (nav_source is null or nav_source in
    ('none', 'cross-site', 'same-site', 'same-origin', 'unknown'))
  not valid;

create index if not exists page_views_nav_source_idx
  on public.page_views (nav_source)
  where nav_source is not null;
