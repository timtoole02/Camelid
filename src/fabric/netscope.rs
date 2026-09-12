//! The address space a scan may cover, and the names it may believe.
//!
//! Two jobs, both of which exist to bound what discovery does to a network it
//! was pointed at.
//!
//! **Ranges are refused, never truncated.** A range outside the private space,
//! or larger than the cap, is turned down with its own size named. Scanning the
//! first 1024 addresses of a /16 and reporting a result would leave an operator
//! believing they had covered a range they had not.
//!
//! **Names are proven, and never on the node resolver.** `getaddrinfo` has no
//! portable cancellation, so a stuck lookup holds its worker for the whole
//! deadline. Live placement resolves node names on a four-worker pool; a scan
//! queueing 254 lookups onto it would stall the requests this proxy exists to
//! serve, so discovery owns a pool of its own.

use std::net::{IpAddr, Ipv4Addr, SocketAddr, ToSocketAddrs};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Serialize;

use super::cancel::Cancel;
use super::http::{self, HttpError, ResolverPool};

/// Every range a scan may cover. There is no override in this phase.
///
/// RFC 1918, carrier-grade NAT, link-local and loopback. A public range is
/// refused: this tool sends traffic to machines an operator names, and naming
/// a public /24 is far more often a typo than an intent.
pub(crate) const ALLOWED_RANGES: [(Ipv4Addr, u8); 6] = [
    (Ipv4Addr::new(10, 0, 0, 0), 8),
    (Ipv4Addr::new(172, 16, 0, 0), 12),
    (Ipv4Addr::new(192, 168, 0, 0), 16),
    (Ipv4Addr::new(100, 64, 0, 0), 10),
    (Ipv4Addr::new(169, 254, 0, 0), 16),
    (Ipv4Addr::new(127, 0, 0, 0), 8),
];

/// The allowed ranges as an operator writes them, for messages and for the
/// policy route.
pub(crate) fn allowed_ranges() -> Vec<String> {
    ALLOWED_RANGES
        .iter()
        .map(|(address, prefix)| Cidr::new(*address, *prefix).to_string())
        .collect()
}

/// Most addresses one scan may cover, across every range it was given.
///
/// A /24 is 254, so a home network fits several times over. A /16 does not, and
/// is refused rather than cut down to this.
pub(crate) const MAX_ADDRESSES: usize = 1024;

/// Most ports one scan may cover.
pub(crate) const MAX_PORTS: usize = 8;

/// How long any one name lookup may take.
pub(crate) const LOOKUP_DEADLINE: Duration = Duration::from_millis(1500);

/// Workers on discovery's own resolver, and how many lookups may queue.
const DISCOVERY_RESOLVER_WORKERS: usize = 8;
const DISCOVERY_RESOLVER_QUEUE: usize = 32;

/// Why a range was refused, before any socket was opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RangeRefusal {
    NotIpv4(String),
    Malformed(String),
    NotPrivate(String),
    TooLarge { addresses: usize, limit: usize },
}

impl std::fmt::Display for RangeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotIpv4(raw) => write!(
                f,
                "`{raw}` is not an IPv4 range; this build sweeps IPv4 only. Name an IPv6 \
                 machine with --host instead"
            ),
            Self::Malformed(raw) => write!(f, "`{raw}` is not a range of the form 100.64.0.0/24"),
            Self::NotPrivate(raw) => write!(
                f,
                "`{raw}` is outside the private address space; this build scans only {}",
                allowed_ranges().join(", ")
            ),
            Self::TooLarge { addresses, limit } => write!(
                f,
                "that range holds {addresses} addresses and this build scans at most {limit} in \
                 one run; it is refused rather than cut short, so nobody believes a range was \
                 covered that was not"
            ),
        }
    }
}

impl RangeRefusal {
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::TooLarge { .. } => "scope_too_large",
            _ => "scope_refused",
        }
    }
}

/// An IPv4 range, already masked to its network address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Cidr {
    base: u32,
    prefix: u8,
}

impl std::fmt::Display for Cidr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", Ipv4Addr::from(self.base), self.prefix)
    }
}

impl Cidr {
    fn new(address: Ipv4Addr, prefix: u8) -> Self {
        let raw = u32::from(address);
        let mask = mask_of(prefix);
        Self {
            base: raw & mask,
            prefix,
        }
    }

    /// How many addresses this range would actually be probed on.
    ///
    /// Network and broadcast are excluded, except on a /31, where RFC 3021
    /// gives both addresses to hosts, and on a /32, which is one host.
    pub(crate) fn host_count(&self) -> usize {
        match self.prefix {
            32 => 1,
            31 => 2,
            prefix => (1_usize << (32 - u32::from(prefix))) - 2,
        }
    }

    pub(crate) fn hosts(&self) -> Vec<Ipv4Addr> {
        let size = 1_u64 << (32 - u32::from(self.prefix));
        let (first, last) = if self.prefix >= 31 {
            (0, size - 1)
        } else {
            (1, size - 2)
        };
        (first..=last)
            .map(|offset| Ipv4Addr::from(self.base.wrapping_add(offset as u32)))
            .collect()
    }

    fn contains(&self, other: &Cidr) -> bool {
        other.prefix >= self.prefix && (other.base & mask_of(self.prefix)) == self.base
    }
}

fn mask_of(prefix: u8) -> u32 {
    if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix))
    }
}

/// Parse and check one range. Every refusal happens here, before a socket.
pub(crate) fn checked_range(raw: &str) -> Result<Cidr, RangeRefusal> {
    let trimmed = raw.trim();
    let (address, prefix) = trimmed
        .split_once('/')
        .ok_or_else(|| RangeRefusal::Malformed(trimmed.to_string()))?;
    if address.contains(':') {
        return Err(RangeRefusal::NotIpv4(trimmed.to_string()));
    }
    let address: Ipv4Addr = address
        .parse()
        .map_err(|_| RangeRefusal::Malformed(trimmed.to_string()))?;
    let prefix: u8 = prefix
        .parse()
        .ok()
        .filter(|prefix| *prefix <= 32)
        .ok_or_else(|| RangeRefusal::Malformed(trimmed.to_string()))?;
    let cidr = Cidr::new(address, prefix);

    if !ALLOWED_RANGES
        .iter()
        .any(|(address, prefix)| Cidr::new(*address, *prefix).contains(&cidr))
    {
        return Err(RangeRefusal::NotPrivate(trimmed.to_string()));
    }

    let addresses = cidr.host_count();
    if addresses > MAX_ADDRESSES {
        return Err(RangeRefusal::TooLarge {
            addresses,
            limit: MAX_ADDRESSES,
        });
    }
    Ok(cidr)
}

/// One address this machine holds, and the size of the network it is on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Interface {
    pub(crate) name: String,
    pub(crate) address: IpAddr,
    /// `None` when the platform gave no netmask, which is a fact about the
    /// lookup rather than about the network.
    pub(crate) prefix: Option<u8>,
}

/// A range the operator might mean, and where its shape came from.
///
/// Offered, never scanned. Nothing here opens a socket; the CLI prints it as a
/// command and the panel pre-fills a field with it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Suggestion {
    pub cidr: String,
    pub interface: Option<String>,
    pub address: Option<String>,
    /// `interface_netmask`, `narrowed` or `assumed_24`. A suggestion always
    /// says how its prefix was arrived at, because two of those are guesses.
    pub prefix_source: &'static str,
}

/// Every address this machine holds, as the platform reports them.
///
/// An empty answer means we could not look, not that the machine has no
/// addresses. Callers report that as unknown.
#[cfg(unix)]
pub(crate) fn local_interfaces() -> Vec<Interface> {
    let mut found = Vec::new();
    let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: `getifaddrs` fills `list` with an owned linked list on success,
    // which is freed below on every path out.
    if unsafe { libc::getifaddrs(&mut list) } != 0 {
        return found;
    }
    let mut entry = list;
    while !entry.is_null() {
        // SAFETY: the list is well-formed until the terminating null, and each
        // node is read, never retained.
        let current = unsafe { &*entry };
        entry = current.ifa_next;
        let Some(address) = (unsafe { address_of(current.ifa_addr) }) else {
            continue;
        };
        let name = unsafe { std::ffi::CStr::from_ptr(current.ifa_name) }
            .to_string_lossy()
            .into_owned();
        let prefix = unsafe { address_of(current.ifa_netmask) }.and_then(prefix_of);
        found.push(Interface {
            name,
            address,
            prefix,
        });
    }
    // SAFETY: `list` came from `getifaddrs` and is freed exactly once.
    unsafe { libc::freeifaddrs(list) };
    found
}

/// Interface enumeration is not implemented for this platform, so nothing is
/// claimed about which addresses are this machine's.
#[cfg(not(unix))]
pub(crate) fn local_interfaces() -> Vec<Interface> {
    Vec::new()
}

/// Read one `sockaddr` as an IP address, or `None` for a family we do not read.
///
/// # Safety
/// `raw` must be null or point at a `sockaddr` whose family field is
/// initialised, as every pointer in a `getifaddrs` list is.
#[cfg(unix)]
unsafe fn address_of(raw: *const libc::sockaddr) -> Option<IpAddr> {
    if raw.is_null() {
        return None;
    }
    match i32::from((*raw).sa_family) {
        libc::AF_INET => {
            let v4 = &*(raw as *const libc::sockaddr_in);
            Some(IpAddr::V4(Ipv4Addr::from(u32::from_be(v4.sin_addr.s_addr))))
        }
        libc::AF_INET6 => {
            let v6 = &*(raw as *const libc::sockaddr_in6);
            Some(IpAddr::V6(std::net::Ipv6Addr::from(v6.sin6_addr.s6_addr)))
        }
        _ => None,
    }
}

/// The prefix length a netmask expresses, for a contiguous mask.
#[cfg(unix)]
fn prefix_of(mask: IpAddr) -> Option<u8> {
    let bits: Vec<u8> = match mask {
        IpAddr::V4(mask) => mask.octets().to_vec(),
        IpAddr::V6(mask) => mask.octets().to_vec(),
    };
    let ones: u32 = bits.iter().map(|byte| byte.count_ones()).sum();
    // A mask with holes in it is not a prefix, and guessing one would be
    // inventing a network boundary.
    let expected = bits.len() as u32 * 8 - ones;
    let trailing: u32 = bits
        .iter()
        .rev()
        .map(|byte| byte.trailing_zeros().min(8))
        .take_while(|zeros| *zeros == 8)
        .count() as u32
        * 8;
    let partial = bits
        .iter()
        .rev()
        .find(|byte| **byte != 0)
        .map_or(0, |byte| byte.trailing_zeros());
    if trailing + partial != expected {
        return None;
    }
    u8::try_from(ones).ok()
}

/// Whether this address belongs to the machine running the scan.
///
/// `None` when the platform would not tell us, which is reported as unknown
/// rather than as "another machine".
pub(crate) fn is_this_machine(address: IpAddr) -> Option<bool> {
    if address.is_loopback() {
        return Some(true);
    }
    let interfaces = local_interfaces();
    if interfaces.is_empty() {
        return None;
    }
    Some(
        interfaces
            .iter()
            .any(|interface| interface.address == address),
    )
}

/// Ranges an operator might mean, most specific first.
pub(crate) fn suggestions() -> Vec<Suggestion> {
    let mut found: Vec<Suggestion> = Vec::new();
    for interface in local_interfaces() {
        let IpAddr::V4(address) = interface.address else {
            continue;
        };
        if address.is_loopback() {
            continue;
        }
        let (cidr, source) = match interface.prefix {
            // Wider than a /24: narrowed to the block around this machine, so a
            // suggestion is never larger than the cap.
            Some(prefix) if prefix < 24 => (Cidr::new(address, 24), "narrowed"),
            Some(prefix) => (Cidr::new(address, prefix), "interface_netmask"),
            None => (Cidr::new(address, 24), "assumed_24"),
        };
        if checked_range(&cidr.to_string()).is_err() {
            continue;
        }
        let suggestion = Suggestion {
            cidr: cidr.to_string(),
            interface: Some(interface.name),
            address: Some(address.to_string()),
            prefix_source: source,
        };
        if !found.iter().any(|seen| seen.cidr == suggestion.cidr) {
            found.push(suggestion);
        }
    }
    if found.is_empty() {
        found.extend(routed_address_suggestion());
    }
    found
}

/// The address a packet to the internet would leave from, without sending one.
///
/// A connected UDP socket only records a route; nothing is transmitted. Used
/// where interface enumeration is unavailable, and labelled as the guess it is.
fn routed_address_suggestion() -> Option<Suggestion> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    // TEST-NET-1, which is never routed anywhere.
    socket.connect("192.0.2.1:9").ok()?;
    let IpAddr::V4(address) = socket.local_addr().ok()?.ip() else {
        return None;
    };
    if address.is_loopback() || address.is_unspecified() {
        return None;
    }
    let cidr = Cidr::new(address, 24);
    checked_range(&cidr.to_string()).ok()?;
    Some(Suggestion {
        cidr: cidr.to_string(),
        interface: None,
        address: Some(address.to_string()),
        prefix_source: "assumed_24",
    })
}

/// What a discovery lookup answered.
enum Looked {
    Addrs(Vec<SocketAddr>),
    Name(String),
}

fn discovery_resolver() -> Result<std::sync::mpsc::SyncSender<http::ResolveJob<Looked>>, HttpError> {
    static RESOLVER: OnceLock<Mutex<Option<ResolverPool<Looked>>>> = OnceLock::new();
    let resolver = RESOLVER.get_or_init(|| Mutex::new(None));
    let mut resolver = resolver
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if resolver.is_none() {
        *resolver = Some(
            ResolverPool::new(
                "camelid-discovery-resolver",
                DISCOVERY_RESOLVER_WORKERS,
                DISCOVERY_RESOLVER_QUEUE,
            )
            .map_err(HttpError::Resolve)?,
        );
    }
    Ok(resolver
        .as_ref()
        .expect("resolver initialized above")
        .sender())
}

/// Resolve a host the way the fabric itself would, on discovery's own pool.
///
/// The same `to_socket_addrs` the node path uses, in the same order, so a name
/// proven here reaches the address the fabric's own connect would reach.
pub(crate) fn forward_lookup(
    host: &str,
    port: u16,
    cancel: &Cancel,
) -> Result<Vec<SocketAddr>, HttpError> {
    let bare = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = bare.parse::<IpAddr>() {
        return Ok(vec![SocketAddr::new(ip, port)]);
    }
    let host = bare.to_string();
    let lookup_host = host.clone();
    let answer = http::resolve_with_sender(
        &discovery_resolver()?,
        Instant::now() + LOOKUP_DEADLINE,
        cancel,
        Box::new(move || {
            (lookup_host.as_str(), port)
                .to_socket_addrs()
                .map(|addrs| Looked::Addrs(addrs.collect()))
                .map_err(|error| format!("{host}: {error}"))
        }),
    )?;
    match answer {
        Looked::Addrs(addrs) => Ok(addrs),
        Looked::Name(_) => Err(HttpError::Resolve(
            "the resolver answered a name where addresses were asked for".to_string(),
        )),
    }
}

/// The name the network claims for an address, or `None`.
///
/// `NI_NAMEREQD`, so an address with no record answers nothing rather than its
/// own digits back. The name is only ever a *candidate*: it is the device's own
/// claim on a consumer router, and callers must prove it forward before
/// listing it.
pub(crate) fn reverse_lookup(address: IpAddr, cancel: &Cancel) -> Option<String> {
    let answer = http::resolve_with_sender(
        &discovery_resolver().ok()?,
        Instant::now() + LOOKUP_DEADLINE,
        cancel,
        Box::new(move || reverse_name(address).map(Looked::Name)),
    )
    .ok()?;
    match answer {
        Looked::Name(name) => Some(name),
        Looked::Addrs(_) => None,
    }
}

#[cfg(unix)]
fn reverse_name(address: IpAddr) -> Result<String, String> {
    let mut host = [0_i8; 256];
    let code = match address {
        IpAddr::V4(v4) => {
            // SAFETY: a zeroed sockaddr_in with its family and address set is a
            // complete value, and its length is passed explicitly.
            let mut sockaddr: libc::sockaddr_in = unsafe { std::mem::zeroed() };
            sockaddr.sin_family = libc::AF_INET as libc::sa_family_t;
            sockaddr.sin_addr.s_addr = u32::from(v4).to_be();
            unsafe {
                libc::getnameinfo(
                    std::ptr::addr_of!(sockaddr).cast(),
                    std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
                    host.as_mut_ptr().cast(),
                    host.len() as libc::socklen_t,
                    std::ptr::null_mut(),
                    0,
                    libc::NI_NAMEREQD,
                )
            }
        }
        IpAddr::V6(v6) => {
            // SAFETY: as above, for the v6 shape.
            let mut sockaddr: libc::sockaddr_in6 = unsafe { std::mem::zeroed() };
            sockaddr.sin6_family = libc::AF_INET6 as libc::sa_family_t;
            sockaddr.sin6_addr.s6_addr = v6.octets();
            unsafe {
                libc::getnameinfo(
                    std::ptr::addr_of!(sockaddr).cast(),
                    std::mem::size_of::<libc::sockaddr_in6>() as libc::socklen_t,
                    host.as_mut_ptr().cast(),
                    host.len() as libc::socklen_t,
                    std::ptr::null_mut(),
                    0,
                    libc::NI_NAMEREQD,
                )
            }
        }
    };
    if code != 0 {
        return Err("no reverse DNS record".to_string());
    }
    // SAFETY: on success `getnameinfo` wrote a NUL-terminated name.
    let name = unsafe { std::ffi::CStr::from_ptr(host.as_ptr().cast()) }
        .to_string_lossy()
        .into_owned();
    if name.is_empty() {
        return Err("no reverse DNS record".to_string());
    }
    Ok(name)
}

#[cfg(not(unix))]
fn reverse_name(_address: IpAddr) -> Result<String, String> {
    Err("not available on this platform".to_string())
}

/// Hold every discovery resolver worker and its whole queue, so a test can show
/// that node resolution is unaffected. Returns the flag that releases them.
#[cfg(test)]
pub(crate) fn saturate_discovery_resolver(
) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let release = Arc::new(AtomicBool::new(false));
    for _ in 0..(DISCOVERY_RESOLVER_WORKERS + DISCOVERY_RESOLVER_QUEUE) {
        let held = Arc::clone(&release);
        std::thread::spawn(move || {
            let _ = http::resolve_with_sender(
                &discovery_resolver().expect("pool"),
                Instant::now() + Duration::from_secs(30),
                &Cancel::never(),
                Box::new(move || {
                    while !held.load(Ordering::SeqCst) {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Ok(Looked::Addrs(Vec::new()))
                }),
            );
        });
    }
    release
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The allowlist is the whole rule. Without it a scan is a port scanner
    /// pointed at somebody else's network.
    #[test]
    fn a_public_range_is_refused_before_any_socket() {
        for public in ["8.8.8.0/24", "192.169.0.0/24", "1.1.1.0/28", "172.32.0.0/24"] {
            let refusal = checked_range(public).expect_err("public ranges are refused");
            assert_eq!(
                refusal,
                RangeRefusal::NotPrivate(public.to_string()),
                "{public}"
            );
            let message = refusal.to_string();
            for allowed in allowed_ranges() {
                assert!(message.contains(&allowed), "{message} is missing {allowed}");
            }
        }
        // Built from octets rather than written as literals: a tracked file
        // may not carry RFC1918 addresses, and the coverage is the point.
        for (address, prefix) in ALLOWED_RANGES {
            let inside = Cidr::new(address, prefix.max(24)).to_string();
            checked_range(&inside).unwrap_or_else(|error| panic!("{inside}: {error}"));
        }
    }

    /// Refused with its own size named, never quietly cut down to the cap.
    #[test]
    fn a_range_over_the_address_limit_is_refused_not_truncated() {
        assert_eq!(
            checked_range("100.64.0.0/16").expect_err("too large"),
            RangeRefusal::TooLarge {
                addresses: 65534,
                limit: MAX_ADDRESSES,
            }
        );
        let message = checked_range("100.64.0.0/10").expect_err("too large").to_string();
        assert!(message.contains("refused rather than cut short"), "{message}");
        // The largest range that still fits, so the cap is a boundary rather
        // than a vague discouragement.
        assert_eq!(checked_range("100.64.4.0/22").expect("fits").host_count(), 1022);
    }

    #[test]
    fn network_and_broadcast_addresses_are_not_probed() {
        let slash24 = checked_range("100.64.0.0/24").expect("private");
        let hosts = slash24.hosts();
        assert_eq!(hosts.len(), 254);
        assert_eq!(hosts.len(), slash24.host_count());
        assert_eq!(hosts[0], Ipv4Addr::new(100, 64, 0, 1));
        assert_eq!(hosts[253], Ipv4Addr::new(100, 64, 0, 254));
        assert!(!hosts.contains(&Ipv4Addr::new(100, 64, 0, 0)));
        assert!(!hosts.contains(&Ipv4Addr::new(100, 64, 0, 255)));

        // RFC 3021: on a /31 both addresses are hosts, and a /32 is one host.
        let slash31 = checked_range("100.64.0.2/31").expect("private");
        assert_eq!(
            slash31.hosts(),
            vec![Ipv4Addr::new(100, 64, 0, 2), Ipv4Addr::new(100, 64, 0, 3)]
        );
        let slash32 = checked_range("100.64.0.37/32").expect("private");
        assert_eq!(slash32.hosts(), vec![Ipv4Addr::new(100, 64, 0, 37)]);
    }

    #[test]
    fn an_ipv6_range_is_refused_with_advice() {
        let refusal = checked_range("fd00::/64").expect_err("IPv6 is refused");
        assert_eq!(refusal, RangeRefusal::NotIpv4("fd00::/64".to_string()));
        assert!(refusal.to_string().contains("--host"), "{refusal}");
        assert!(matches!(
            checked_range("100.64.0.0"),
            Err(RangeRefusal::Malformed(_))
        ));
        assert!(matches!(
            checked_range("100.64.0.0/33"),
            Err(RangeRefusal::Malformed(_))
        ));
    }

    /// An operator writes the address they know, not the network address.
    #[test]
    fn a_range_written_from_a_host_address_means_the_network_it_is_on() {
        let from_host = checked_range("100.64.0.37/28").expect("private");
        assert_eq!(from_host.to_string(), "100.64.0.32/28");
        assert_eq!(from_host.host_count(), 14);
        assert!(from_host.hosts().contains(&Ipv4Addr::new(100, 64, 0, 37)));
    }

    #[test]
    fn a_netmask_with_holes_in_it_is_not_a_prefix() {
        assert_eq!(prefix_of(IpAddr::V4(Ipv4Addr::new(255, 255, 255, 0))), Some(24));
        assert_eq!(prefix_of(IpAddr::V4(Ipv4Addr::new(255, 255, 240, 0))), Some(20));
        assert_eq!(prefix_of(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0))), Some(0));
        assert_eq!(prefix_of(IpAddr::V4(Ipv4Addr::new(255, 0, 255, 0))), None);
    }

    /// Loopback is this machine on every platform. Anything else depends on
    /// enumeration, which may be unavailable — and is then unknown, never false.
    #[test]
    fn loopback_is_always_this_machine() {
        assert_eq!(is_this_machine("127.0.0.1".parse().expect("ip")), Some(true));
        assert_eq!(is_this_machine("::1".parse().expect("ip")), Some(true));
        // A documentation address is nobody's interface.
        if !local_interfaces().is_empty() {
            assert_eq!(
                is_this_machine("192.0.2.99".parse().expect("ip")),
                Some(false)
            );
        }
    }

    /// A suggestion is offered, not scanned, and always says where it came from.
    #[test]
    fn every_suggestion_names_how_its_prefix_was_arrived_at() {
        for suggestion in suggestions() {
            assert!(
                ["interface_netmask", "narrowed", "assumed_24"]
                    .contains(&suggestion.prefix_source),
                "{suggestion:?}"
            );
            checked_range(&suggestion.cidr)
                .unwrap_or_else(|error| panic!("{}: {error}", suggestion.cidr));
        }
    }

    #[test]
    fn an_address_literal_needs_no_lookup_at_all() {
        assert_eq!(
            forward_lookup("127.0.0.1", 8181, &Cancel::never()).expect("literal"),
            vec![SocketAddr::from(([127, 0, 0, 1], 8181))]
        );
        assert_eq!(
            forward_lookup("[::1]", 8181, &Cancel::never()).expect("literal"),
            vec!["[::1]:8181".parse::<SocketAddr>().expect("addr")]
        );
    }
}
