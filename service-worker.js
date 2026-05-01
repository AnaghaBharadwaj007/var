self.addEventListener('install', () => {
  console.log('Service Worker Installed');
});

self.addEventListener('fetch', (event) => {
  // basic pass-through (no caching needed for now)
});