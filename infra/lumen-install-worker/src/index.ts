const GITHUB_RELEASE_BASE =
  "https://github.com/EdwinZhanCN/Lumen-Hub/releases/latest/download";

const INSTALLERS: Record<string, { fileName: string; contentType: string }> = {
  "/lumen/install.sh": {
    fileName: "install.sh",
    contentType: "application/x-sh; charset=utf-8",
  },
  "/lumen/install.ps1": {
    fileName: "install.ps1",
    contentType: "text/plain; charset=utf-8",
  },
};

export default {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const installer = INSTALLERS[url.pathname];

    if (!installer) {
      return new Response("Not found\n", { status: 404 });
    }

    const upstream = await fetch(
      `${GITHUB_RELEASE_BASE}/${installer.fileName}`,
      {
        headers: {
          "User-Agent": "lumen-install-worker",
        },
      },
    );

    if (!upstream.ok || !upstream.body) {
      return new Response(`Failed to fetch ${installer.fileName}\n`, {
        status: upstream.status,
      });
    }

    const headers = new Headers();
    headers.set("Content-Type", installer.contentType);
    headers.set("Cache-Control", "public, max-age=300");
    headers.set("X-Content-Type-Options", "nosniff");

    return new Response(upstream.body, {
      status: upstream.status,
      headers,
    });
  },
};
