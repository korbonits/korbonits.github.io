-- Run this SQL in the Supabase SQL editor to set up analytics.
-- Project: korbonits.com page view tracking

create table if not exists page_views (
  id       bigint generated always as identity primary key,
  path     text        not null,
  referrer text,
  user_agent text,
  country  char(2),
  created_at timestamptz not null default now()
);

-- Index for querying by page (top pages report)
create index if not exists page_views_path_idx
  on page_views (path);

-- Index for querying by time (views today / last 7d / last 30d)
create index if not exists page_views_created_at_idx
  on page_views (created_at desc);

-- Allow the anon key to insert rows (called from the Netlify function)
-- and read rows (called from the Astro analytics page at build time).
-- If you prefer stricter access, replace anon with a dedicated service role
-- and use the service_role key instead of the anon key in SUPABASE_ANON_KEY.
alter table page_views enable row level security;

create policy "Allow anon insert"
  on page_views for insert
  to anon
  with check (true);

create policy "Allow anon select"
  on page_views for select
  to anon
  using (true);
